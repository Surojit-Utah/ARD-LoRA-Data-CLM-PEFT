from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
import torch
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, Subset
import json
import os
import gc
from pathlib import Path

# Import uncertainty evaluator
try:
    from evaluate.uncertainty_metrics import UncertaintyEvaluator, print_evaluation_results
except ImportError:
    print("[WARNING] Could not import uncertainty evaluator. Please ensure evaluate module is available.")
    UncertaintyEvaluator = None

# Import plotting utilities if available
try:
    from utils.plot import plot_mean_encodings, diag_axis_splom
except ImportError:
    print("[INFO] Plotting utilities not available. Latent plotting will be disabled.")
    plot_mean_encodings = None


class ARDCLMTrainer(Trainer):
    """Enhanced ARD Trainer with uncertainty evaluation after each epoch."""
    
    def __init__(self, *args, beta=0.01, ard_eval_dataset=None, uncertainty_eval_samples=1000, 
                 n_bins=15, output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.ard_eval_dataset = ard_eval_dataset  # Dataset for ARD prior estimation
        self.uncertainty_eval_samples = uncertainty_eval_samples
        self.n_bins = n_bins
        self.uncertainty_evaluator = UncertaintyEvaluator(n_bins=n_bins) if UncertaintyEvaluator else None
        self.uncertainty_results = []  # Store results across epochs
        self.output_dir = output_dir or self.args.output_dir
        
        # Track loss components for logging
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Enhanced compute_loss with KL regularization from ProbLoRA layers."""
        # Extract labels and handle causal LM loss computation
        labels = inputs.get('labels')
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute base loss (causal LM loss)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            ce_loss = outputs.loss
        else:
            # Manual causal LM loss computation
            logits = outputs.logits
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                ce_loss = loss_fct(shift_logits, shift_labels)
            else:
                ce_loss = torch.tensor(0.0, device=outputs.logits.device)

        # Compute KL divergence from ProbLoRA layers
        kl = 0.0
        
        # Method 1: Try to get KL from module attributes (if available)
        for module in model.modules():
            if hasattr(module, 'kl_divergence'):
                try:
                    module_kl = module.kl_divergence()
                    if torch.is_tensor(module_kl):
                        kl += module_kl
                except Exception as e:
                    # Skip modules that fail KL computation
                    continue
        
        # Method 2: Enhanced KL computation for LLaMA ProbLoRA layers
        if kl == 0.0:  # Fallback if standard method didn't work
            try:
                # For LLaMA models with ProbLoRA layers
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    for layer_idx, layer in enumerate(model.model.layers):
                        # Check attention layers
                        if hasattr(layer, 'self_attn'):
                            attn = layer.self_attn
                            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                                if hasattr(attn, proj_name):
                                    proj = getattr(attn, proj_name)
                                    if hasattr(proj, 'A') and hasattr(proj, 'B'):  # ProbLoRA layer
                                        try:
                                            if hasattr(proj, 'kl_divergence'):
                                                kl += proj.kl_divergence()
                                            elif hasattr(proj, 'log_sigma_A') and hasattr(proj, 'log_sigma_B'):
                                                # Manual KL computation
                                                kl += self._compute_problora_kl(proj)
                                        except Exception:
                                            continue
                        
                        # Check MLP layers
                        if hasattr(layer, 'mlp'):
                            mlp = layer.mlp
                            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                                if hasattr(mlp, proj_name):
                                    proj = getattr(mlp, proj_name)
                                    if hasattr(proj, 'A') and hasattr(proj, 'B'):  # ProbLoRA layer
                                        try:
                                            if hasattr(proj, 'kl_divergence'):
                                                kl += proj.kl_divergence()
                                            elif hasattr(proj, 'log_sigma_A') and hasattr(proj, 'log_sigma_B'):
                                                # Manual KL computation
                                                kl += self._compute_problora_kl(proj)
                                        except Exception:
                                            continue
            except Exception as e:
                # If enhanced method fails, use basic approach
                pass
        
        # Ensure KL is a tensor
        if not torch.is_tensor(kl):
            kl = torch.tensor(kl, device=ce_loss.device)
        
        # Total loss with KL regularization
        total_loss = ce_loss + self.beta * kl
        
        # Store loss components for logging (convert to float for consistency)
        self.last_ce_loss = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
        self.last_kl_loss = (kl.item() if torch.is_tensor(kl) else float(kl))
        self.last_total_loss = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
        
        # Memory cleanup after computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def _compute_problora_kl(self, layer):
        """Manual KL divergence computation for ProbLoRA layers."""
        try:
            # KL divergence for variational parameters
            # KL(q(theta) || p(theta)) for Gaussian priors
            kl_A = 0.0
            kl_B = 0.0
            
            if hasattr(layer, 'log_sigma_A') and hasattr(layer, 'mu_A'):
                # KL for A parameters: KL(N(mu_A, sigma_A^2) || N(0, prior_var))
                prior_var = getattr(layer, 'prior_var', 1.0)
                sigma_A_sq = torch.exp(2 * layer.log_sigma_A)
                mu_A_sq = layer.mu_A ** 2
                
                kl_A = 0.5 * torch.sum(
                    mu_A_sq / prior_var + sigma_A_sq / prior_var - 1.0 - 2 * layer.log_sigma_A + np.log(prior_var)
                )
            
            if hasattr(layer, 'log_sigma_B') and hasattr(layer, 'mu_B'):
                # KL for B parameters
                prior_var = getattr(layer, 'prior_var', 1.0)
                sigma_B_sq = torch.exp(2 * layer.log_sigma_B)
                mu_B_sq = layer.mu_B ** 2
                
                kl_B = 0.5 * torch.sum(
                    mu_B_sq / prior_var + sigma_B_sq / prior_var - 1.0 - 2 * layer.log_sigma_B + np.log(prior_var)
                )
            
            return kl_A + kl_B
        except Exception:
            return torch.tensor(0.0, device=layer.mu_A.device if hasattr(layer, 'mu_A') else 'cpu')
    
    def evaluate_uncertainty(self, eval_dataset=None, num_samples=None) -> Optional[Dict[str, float]]:
        """Evaluate model uncertainty using ACC, ECE, and NLL metrics."""
        if self.uncertainty_evaluator is None:
            print("[WARNING] Uncertainty evaluator not available")
            return None
            
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        if eval_dataset is None:
            print("[WARNING] No evaluation dataset available for uncertainty evaluation")
            return None
            
        num_samples = num_samples or self.uncertainty_eval_samples
        
        print(f"\n🔄 Starting uncertainty evaluation (samples: {num_samples})...")
        
        # Memory optimization: Reduce batch size for evaluation
        original_eval_batch_size = self.args.per_device_eval_batch_size
        if torch.cuda.is_available():
            # Reduce eval batch size to save memory
            memory_optimized_batch_size = max(1, original_eval_batch_size // 2)
            self.args.per_device_eval_batch_size = memory_optimized_batch_size
            print(f"[MEMORY] Reducing eval batch size from {original_eval_batch_size} to {memory_optimized_batch_size}")
        
        # Create evaluation dataloader with smaller batch size
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        self.model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, batch in enumerate(eval_dataloader):
                if sample_count >= num_samples:
                    break
                
                # Move batch to device
                inputs = self._prepare_inputs(batch)
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                labels = inputs.get('labels')
                
                if labels is None:
                    continue
                
                # For causal LM: shift predictions and labels
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten tokens
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Keep only non-masked tokens (labels != -100)
                valid_mask = shift_labels != -100
                valid_logits = shift_logits[valid_mask]
                valid_labels = shift_labels[valid_mask]
                
                if len(valid_labels) > 0:
                    # Convert logits to probabilities
                    probs = torch.softmax(valid_logits, dim=-1)
                    
                    # Sample subset if too many tokens
                    max_tokens_per_batch = min(200, num_samples - sample_count)
                    if len(valid_labels) > max_tokens_per_batch:
                        indices = torch.randperm(len(valid_labels))[:max_tokens_per_batch]
                        valid_labels = valid_labels[indices]
                        probs = probs[indices]
                    
                    # Collect predictions
                    all_labels.extend(valid_labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    sample_count += len(valid_labels)
                
                if batch_idx % 10 == 0:
                    print(f"   Processed {batch_idx + 1} batches, {sample_count} predictions")
        
        if len(all_labels) == 0:
            print("[WARNING] No valid predictions found for uncertainty evaluation")
            return None
            
        print(f"✅ Uncertainty evaluation completed on {len(all_labels)} predictions")
        
        # Restore original batch size
        if torch.cuda.is_available():
            self.args.per_device_eval_batch_size = original_eval_batch_size
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_probs = np.array(all_probs)
        
        # Compute uncertainty metrics
        metrics = self.uncertainty_evaluator.evaluate_predictions(y_true, y_probs)
        
        return metrics
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch to run uncertainty evaluation and log metrics."""
        super().on_epoch_end(args, state, control, model=model, **kwargs)
        
        # Log training loss components to TensorBoard
        if self.args.report_to and 'tensorboard' in self.args.report_to:
            training_metrics = {
                'train/ce_loss': self.last_ce_loss,
                'train/kl_loss': self.last_kl_loss, 
                'train/total_loss': self.last_total_loss,
                'train/kl_beta': self.beta
            }
            self.log(training_metrics)
            print(f"\n📊 Training Loss Components (Epoch {state.epoch}):")
            print(f"   CE Loss: {self.last_ce_loss:.4f}")
            print(f"   KL Loss: {self.last_kl_loss:.4f}")
            print(f"   Total Loss: {self.last_total_loss:.4f}")
            print(f"   KL Beta: {self.beta:.4f}")
        
        # Run evaluation and log eval losses
        if self.eval_dataset is not None:
            print(f"\n📊 Running evaluation after epoch {state.epoch}...")
            
            # Get evaluation metrics including loss
            eval_results = self.evaluate()
            
            # Extract eval loss components if available
            eval_loss = eval_results.get('eval_loss', 0.0)
            
            # Run one forward pass on eval set to get loss components
            eval_ce_loss, eval_kl_loss = self._compute_eval_loss_components(model)
            
            # Log evaluation loss components to TensorBoard
            if self.args.report_to and 'tensorboard' in self.args.report_to:
                eval_metrics = {
                    'eval/ce_loss': eval_ce_loss,
                    'eval/kl_loss': eval_kl_loss,
                    'eval/total_loss': eval_loss
                }
                self.log(eval_metrics)
                
            print(f"\n📊 Evaluation Loss Components (Epoch {state.epoch}):")
            print(f"   CE Loss: {eval_ce_loss:.4f}")
            print(f"   KL Loss: {eval_kl_loss:.4f}")
            print(f"   Total Loss: {eval_loss:.4f}")
            
            # Run uncertainty evaluation
            print(f"\n📊 Running uncertainty evaluation after epoch {state.epoch}...")
            metrics = self.evaluate_uncertainty()
            
            if metrics is not None:
                # Add epoch information
                metrics['epoch'] = state.epoch
                metrics['global_step'] = state.global_step
                
                # Store results
                self.uncertainty_results.append(metrics)
                
                # Print formatted results
                print(f"\n📈 Epoch {state.epoch} Uncertainty Results:")
                print(f"   Accuracy (ACC): {metrics['accuracy']:.4f}")
                print(f"   Expected Calibration Error (ECE): {metrics['ece']:.4f}")
                print(f"   Negative Log-Likelihood (NLL): {metrics['nll']:.4f}")
                
                # Log uncertainty metrics to tensorboard
                if self.args.report_to and 'tensorboard' in self.args.report_to:
                    uncertainty_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            uncertainty_metrics[f"uncertainty/{key}"] = value
                    self.log(uncertainty_metrics)
                
                # Save results to file
                self._save_uncertainty_results()
            
            # Run ARD prior estimation if dataset available
            if self.ard_eval_dataset is not None:
                print(f"\n🔄 Running ARD prior estimation after epoch {state.epoch}...")
                try:
                    estimate_ard_priors_clm(model, self.ard_eval_dataset, model.device)
                    print("✅ ARD prior estimation completed")
                except Exception as e:
                    print(f"⚠️ ARD prior estimation failed: {e}")
    
    def _compute_eval_loss_components(self, model):
        """Compute CE and KL loss components on evaluation dataset."""
        try:
            # Set model to eval mode
            model.eval()
            
            # Get a batch from eval dataset
            eval_dataloader = self.get_eval_dataloader()
            batch = next(iter(eval_dataloader))
            
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            with torch.no_grad():
                # Forward pass to get loss components
                outputs = model(**batch)
                
                # Get CE loss
                labels = batch.get('labels')
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    ce_loss = outputs.loss
                else:
                    # Manual causal LM loss computation
                    logits = outputs.logits
                    if labels is not None:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss()
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)
                        shift_labels = shift_labels.to(shift_logits.device)
                        ce_loss = loss_fct(shift_logits, shift_labels)
                    else:
                        ce_loss = torch.tensor(0.0, device=outputs.logits.device)
                
                # Compute KL divergence
                kl = 0.0
                for module in model.modules():
                    if hasattr(module, 'kl_divergence'):
                        try:
                            module_kl = module.kl_divergence()
                            if torch.is_tensor(module_kl):
                                kl += module_kl
                        except Exception:
                            continue
                
                # Enhanced KL computation fallback
                if kl == 0.0:
                    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        for layer in model.model.layers:
                            if hasattr(layer, 'self_attn'):
                                attn = layer.self_attn
                                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                                    if hasattr(attn, proj_name):
                                        proj = getattr(attn, proj_name)
                                        if hasattr(proj, 'kl_divergence'):
                                            try:
                                                kl += proj.kl_divergence()
                                            except Exception:
                                                continue
                
                # Convert to float values
                ce_loss_val = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
                kl_loss_val = kl.item() if torch.is_tensor(kl) else float(kl)
                
                return ce_loss_val, kl_loss_val
                
        except Exception as e:
            print(f"⚠️ Failed to compute eval loss components: {e}")
            return 0.0, 0.0
        finally:
            # Set model back to train mode
            model.train()

    def _save_uncertainty_results(self):
        """Save uncertainty evaluation results to JSON file."""
        if not self.uncertainty_results:
            return
            
        results_path = Path(self.output_dir) / "uncertainty_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.uncertainty_results, f, indent=2)
            
        print(f"💾 Uncertainty results saved to: {results_path}")


class PriorEstimationCallback(TrainerCallback):
    """Callback to estimate ARD priors at the beginning of each epoch."""
    
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Estimate ARD priors using held-out data at the beginning of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[PriorEstimationCallback] No trainer reference found on model")
            return
            
        if not hasattr(trainer, "ard_eval_dataset") or trainer.ard_eval_dataset is None:
            print("[PriorEstimationCallback] No ARD evaluation dataset found for this epoch")
            return
        
        print(f"[PriorEstimationCallback] Estimating ARD priors at epoch {int(state.epoch)}...")
        
        try:
            # Use the enhanced ARD prior estimation
            estimate_ard_priors_clm(model, trainer.ard_eval_dataset, self.device)
            print("[PriorEstimationCallback] ARD prior estimation completed")
        except Exception as e:
            print(f"[PriorEstimationCallback] Failed to estimate priors: {e}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LatentPlotCallback(TrainerCallback):
    """Callback to plot latent encodings during training."""
    
    def __init__(self, device, output_dir, start_epoch=2, interval=2):
        super().__init__()
        self.device = device
        self.output_dir = Path(output_dir)
        self.start_epoch = start_epoch
        self.interval = interval
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Plot latent encodings at specified intervals."""
        current_epoch = int(state.epoch)
        
        # Check if we should plot this epoch
        if current_epoch < self.start_epoch or (current_epoch - self.start_epoch) % self.interval != 0:
            return
        
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None or not hasattr(trainer, "ard_eval_dataset") or trainer.ard_eval_dataset is None:
            print("[LatentPlotCallback] No ARD evaluation dataset found for plotting")
            return
        
        if plot_mean_encodings is None:
            print("[LatentPlotCallback] Plotting utilities not available")
            return
        
        print(f"[LatentPlotCallback] Plotting latent encodings at epoch {current_epoch}...")
        
        try:
            # Create plots directory
            plot_dir = self.output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Create data loader for plotting
            from torch.utils.data import DataLoader
            plot_dataloader = DataLoader(trainer.ard_eval_dataset, batch_size=16, shuffle=False)
            
            # Generate plots
            plot_mean_encodings(model, plot_dataloader, self.device, str(plot_dir), epoch=current_epoch)
            print(f"[LatentPlotCallback] Plots saved to {plot_dir}")
        except Exception as e:
            print(f"[LatentPlotCallback] Failed to generate plots: {e}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HeldoutResampleCallback(TrainerCallback):
    """Callback to resample held-out data at the beginning of each epoch."""
    
    def __init__(self, train_ds, val_ds, ard_prior_samples, batch_size, tokenizer=None):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.ard_prior_samples = min(ard_prior_samples, len(train_ds) if train_ds else 0)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
    
    def _create_data_loaders(self):
        """Create new data loaders with resampled held-out data."""
        if self.train_ds is None:
            return None, None, None
        
        # Randomly sample indices for held-out data
        total_train = len(self.train_ds)
        perm = np.random.permutation(total_train)
        held_idx = perm[:self.ard_prior_samples].tolist()
        remaining_train_idx = perm[self.ard_prior_samples:].tolist()
        
        # Create subset datasets
        held_ds = Subset(self.train_ds, held_idx)
        remaining_train_ds = Subset(self.train_ds, remaining_train_idx)
        
        # Create data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            remaining_train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )
        heldout_loader = DataLoader(
            held_ds, 
            batch_size=self.batch_size,
            collate_fn=data_collator
        )
        val_loader = None
        if self.val_ds is not None:
            val_loader = DataLoader(
                self.val_ds, 
                batch_size=self.batch_size,
                collate_fn=data_collator
            )
        
        return train_loader, heldout_loader, val_loader
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Resample held-out data at the beginning of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[HeldoutResampleCallback] No trainer reference found")
            return
        
        try:
            train_loader, heldout_loader, val_loader = self._create_data_loaders()
            
            if heldout_loader is not None:
                # Update trainer's held-out dataset
                trainer.ard_eval_dataset = heldout_loader.dataset
                print(f"[HeldoutResampleCallback] Epoch {int(state.epoch)} → new held-out set with {len(heldout_loader.dataset)} samples")
            
            # Note: We don't update the main training dataloader here as it would interfere
            # with the Trainer's internal state. The held-out resampling is mainly for
            # ARD prior estimation.
            
        except Exception as e:
            print(f"[HeldoutResampleCallback] Failed to resample held-out data: {e}")


def split_validation_dataset(val_dataset, ard_prior_ratio=0.5, seed=42, train_dataset=None, train_split_ratio=0.1):
    """
    Split validation dataset into two parts:
    1. ARD prior estimation (first part) - at least 1000 samples
    2. Uncertainty evaluation (second part) - remaining validation data
    
    If val_dataset is None, creates a validation split from train_dataset.
    
    Args:
        val_dataset: Original validation dataset (can be None)
        ard_prior_ratio: Fraction to use for ARD prior estimation (default: 0.5)
        seed: Random seed for reproducible splits
        train_dataset: Training dataset (used if val_dataset is None)
        train_split_ratio: Fraction of training data to use for validation when val_dataset is None (min 10%)
    
    Returns:
        ard_dataset: Dataset for ARD prior estimation (at least 1000 samples)
        uncertainty_dataset: Dataset for uncertainty evaluation (remaining validation data)
    """
    # If no validation dataset, create one from training data
    if val_dataset is None or len(val_dataset) == 0:
        if train_dataset is None or len(train_dataset) == 0:
            print("[WARNING] No validation or training dataset available")
            return None, None
        
        # Ensure validation split is at least 10% of training data
        min_split_ratio = max(0.1, train_split_ratio)
        
        print(f"[INFO] No validation dataset found. Creating validation split from training data ({min_split_ratio:.1%})")
        
        # Create validation split from training data
        total_train = len(train_dataset)
        # Calculate validation size: at least 10% of training data and at least 2000 samples (1000 for ARD + 1000 for eval)
        min_val_size = max(int(total_train * min_split_ratio), 2000)
        val_size = min(min_val_size, total_train // 2)  # Don't use more than half of training data
        
        np.random.seed(seed)
        val_indices = np.random.choice(total_train, val_size, replace=False)
        val_dataset = Subset(train_dataset, val_indices)
        
        print(f"[INFO] Created validation split with {len(val_dataset)} samples from training data ({len(val_dataset)/total_train:.1%} of training data)")
    
    total_size = len(val_dataset)
    if total_size == 0:
        return None, None
    
    # Ensure ARD gets at least 1000 samples, but not more than 80% of validation data
    min_ard_samples = 1000
    max_ard_ratio = 0.8
    
    if total_size < min_ard_samples:
        print(f"[WARNING] Validation dataset has only {total_size} samples, less than required {min_ard_samples} for ARD")
        ard_size = total_size
        uncertainty_size = 0
    else:
        # Calculate ARD size: at least 1000, but respect the ratio and max ratio constraints
        ard_size_by_ratio = int(total_size * ard_prior_ratio)
        ard_size_by_max = int(total_size * max_ard_ratio)
        
        ard_size = max(min_ard_samples, min(ard_size_by_ratio, ard_size_by_max))
        uncertainty_size = total_size - ard_size
    
    # Create indices for splitting
    np.random.seed(seed)
    indices = np.random.permutation(total_size)
    
    ard_indices = indices[:ard_size]
    uncertainty_indices = indices[ard_size:ard_size + uncertainty_size] if uncertainty_size > 0 else []
    
    # Create subset datasets
    ard_dataset = Subset(val_dataset, ard_indices)
    uncertainty_dataset = Subset(val_dataset, uncertainty_indices) if len(uncertainty_indices) > 0 else None
    
    print(f"[INFO] Split validation dataset:")
    print(f"   Total validation samples: {total_size}")
    print(f"   ARD prior estimation: {len(ard_dataset)} samples ({len(ard_dataset)/total_size:.1%})")
    print(f"   Uncertainty evaluation: {len(uncertainty_dataset) if uncertainty_dataset else 0} samples ({len(uncertainty_indices)/total_size:.1%} if uncertainty_dataset else 0)")
    
    return ard_dataset, uncertainty_dataset
    
    # Create subset datasets
    ard_dataset = Subset(val_dataset, ard_indices)
    uncertainty_dataset = Subset(val_dataset, uncertainty_indices) if len(uncertainty_indices) > 0 else None
    
    print(f"[INFO] Split validation dataset:")
    print(f"   Total samples: {total_size}")
    print(f"   ARD prior estimation: {len(ard_dataset)} samples")
    print(f"   Uncertainty evaluation: {len(uncertainty_dataset) if uncertainty_dataset else 0} samples")
    
    return ard_dataset, uncertainty_dataset


def estimate_ard_priors_clm(model, eval_dataset, device, num_samples=1000):
    """
    Estimate ARD priors using a subset of the evaluation dataset.
    This demonstrates the ARD mechanism for automatic relevance determination.
    
    Args:
        model: ARD-LoRA model with ProbLoRA layers
        eval_dataset: Dataset or DataLoader for prior estimation
        device: Device to run computation on
        num_samples: Number of samples to use for estimation
    """
    print(f"[ARD] Estimating priors using {num_samples} samples...")
    
    model.eval()
    
    # Create dataloader if needed
    if not isinstance(eval_dataset, DataLoader):
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=model.config.tokenizer if hasattr(model.config, 'tokenizer') else None,
            mlm=False
        )
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=4, 
            collate_fn=data_collator,
            shuffle=False
        )
    else:
        eval_dataloader = eval_dataset
    
    # Collect statistics from ProbLoRA layers
    layer_stats = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if sample_count >= num_samples:
                break
            
            # Move to device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Collect statistics from ProbLoRA layers
            for name, module in model.named_modules():
                if hasattr(module, 'A') and hasattr(module, 'B'):
                    # This is a ProbLoRA layer
                    if hasattr(module, 'log_sigma_A') and hasattr(module, 'log_sigma_B'):
                        # Compute variance estimates
                        var_A = torch.exp(2 * module.log_sigma_A).mean().item()
                        var_B = torch.exp(2 * module.log_sigma_B).mean().item()
                        
                        layer_stats.append({
                            'layer_name': name,
                            'var_A': var_A,
                            'var_B': var_B,
                            'combined_var': var_A * var_B
                        })
            
            sample_count += inputs['input_ids'].size(0) if 'input_ids' in inputs else 1
            
            if batch_idx % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches, ~{sample_count} samples")
    
    # Analyze collected statistics
    if layer_stats:
        print(f"\n[ARD] Analysis of {len(set(s['layer_name'] for s in layer_stats))} ProbLoRA layers:")
        
        # Group by layer
        layer_groups = {}
        for stat in layer_stats:
            name = stat['layer_name']
            if name not in layer_groups:
                layer_groups[name] = []
            layer_groups[name].append(stat)
        
        # Print statistics
        for layer_name, stats in layer_groups.items():
            avg_var_A = np.mean([s['var_A'] for s in stats])
            avg_var_B = np.mean([s['var_B'] for s in stats])
            avg_combined = np.mean([s['combined_var'] for s in stats])
            
            print(f"   {layer_name}: Var(A)={avg_var_A:.6f}, Var(B)={avg_var_B:.6f}, Combined={avg_combined:.6f}")
            
            # ARD relevance determination
            relevance = "High" if avg_combined > 0.01 else "Medium" if avg_combined > 0.001 else "Low"
            print(f"     → Estimated relevance: {relevance}")
    else:
        print("[ARD] No ProbLoRA layers found for prior estimation")
    
    model.train()  # Return to training mode
    
    return layer_stats


def optimize_memory_settings():
    """
    Apply memory optimization settings to reduce CUDA OOM errors.
    Call this before training starts.
    """
    print("[MEMORY] Applying memory optimization settings...")
    
    # Set CUDA memory allocation configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Reduce memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Set memory fraction to leave some memory for system
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            print("[MEMORY] Set GPU memory fraction to 90%")
        except:
            print("[MEMORY] Could not set memory fraction")
        
        # Print current memory status
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        cached_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"[MEMORY] GPU Total: {total_memory:.2f} GB")
        print(f"[MEMORY] GPU Allocated: {allocated_memory:.2f} GB")
        print(f"[MEMORY] GPU Cached: {cached_memory:.2f} GB")
        print(f"[MEMORY] GPU Free: {total_memory - cached_memory:.2f} GB")
    
    # Set environment variables for memory optimization
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Reduce tokenizer memory usage
    
    print("[MEMORY] Memory optimization settings applied")


def emergency_memory_cleanup():
    """
    Emergency memory cleanup function to call when OOM occurs.
    """
    print("[MEMORY] Emergency memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[MEMORY] Emergency cleanup completed")


def create_ard_callbacks(device, output_dir, train_ds=None, val_ds=None, 
                        ard_prior_samples=1000, batch_size=4, tokenizer=None,
                        enable_plotting=True, enable_resampling=True,
                        plot_start_epoch=2, plot_interval=2):
    """Create standard ARD training callbacks.
    
    Args:
        device: Device for computation
        output_dir: Output directory for plots and logs
        train_ds: Training dataset (for resampling)
        val_ds: Validation dataset
        ard_prior_samples: Number of samples for ARD prior estimation
        batch_size: Batch size for data loaders
        tokenizer: Tokenizer for data collation
        enable_plotting: Whether to enable latent plotting
        enable_resampling: Whether to enable held-out resampling
        plot_start_epoch: Epoch to start plotting
        plot_interval: Interval between plots
    
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Always add prior estimation callback
    callbacks.append(PriorEstimationCallback(device))
    
    # Add plotting callback if enabled and utilities available
    if enable_plotting and plot_mean_encodings is not None:
        callbacks.append(LatentPlotCallback(
            device=device,
            output_dir=output_dir,
            start_epoch=plot_start_epoch,
            interval=plot_interval
        ))
    
    # Add resampling callback if enabled and datasets available
    if enable_resampling and train_ds is not None:
        callbacks.append(HeldoutResampleCallback(
            train_ds=train_ds,
            val_ds=val_ds,
            ard_prior_samples=ard_prior_samples,
            batch_size=batch_size,
            tokenizer=tokenizer
        ))
    
    return callbacks


def build_clm_trainer(model, tokenizer, train_dataset, eval_dataset, cfg, output_dir, 
                     ard_prior_ratio=0.5, enable_callbacks=True, tb_log_dir=None):
    """Build enhanced CLM trainer with uncertainty evaluation, ARD callbacks, and prior estimation."""
    
    # Split validation dataset for ARD and uncertainty evaluation
    # Pass train_dataset so we can create validation split if needed
    ard_dataset, uncertainty_dataset = split_validation_dataset(
        eval_dataset, 
        ard_prior_ratio=ard_prior_ratio,
        train_dataset=train_dataset,
        train_split_ratio=cfg.get("validation_split_ratio", 0.1)  # Use 10% of training data by default (minimum)
    )
    
    # Use TensorBoard log directory if provided, otherwise use default
    logging_dir = tb_log_dir or cfg.get("logging_dir") or f"{output_dir}/tensorboard_logs"
    
    # Training arguments with enhanced configuration and memory optimization
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,  # Explicit TensorBoard logging directory
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=max(1, cfg.get("batch_size", 1)),  # Smaller eval batch to save memory
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),  # Increase to compensate for smaller batch
        num_train_epochs=cfg.get("train_epochs", 1),
        fp16=bool(cfg.get("fp16", True)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=cfg.get("logging_steps", 50),
        save_strategy="epoch",  # Save after each epoch
        eval_strategy="epoch" if uncertainty_dataset else "no",  # Evaluate after each epoch
        save_steps=500,
        eval_steps=None,  # Use epoch-based evaluation
        load_best_model_at_end=bool(uncertainty_dataset),
        metric_for_best_model="eval_loss" if uncertainty_dataset else None,
        report_to=cfg.get("report_to", ["tensorboard"]) if cfg.get("report_to") else None,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Reduce to 0 to save memory
        # Memory optimization settings
        max_grad_norm=1.0,  # Gradient clipping
        optim="adamw_torch",  # Use torch AdamW for better memory management
        # Enable gradient checkpointing if model supports it
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create the enhanced trainer
    trainer = ARDCLMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=uncertainty_dataset,  # Use uncertainty dataset for standard evaluation
        data_collator=data_collator,
        tokenizer=tokenizer,
        beta=float(cfg.get("beta", 0.01)),
        ard_eval_dataset=ard_dataset,  # Separate dataset for ARD prior estimation
        uncertainty_eval_samples=cfg.get("uncertainty_eval_samples", 1000),
        n_bins=cfg.get("uncertainty_n_bins", 15),
        output_dir=output_dir
    )
    
    # Set trainer reference on model for callbacks
    model.trainer = trainer
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("[MEMORY] Enabled gradient checkpointing")
    
    # Memory optimization: Clear cache and show memory status
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[MEMORY] GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        print(f"[MEMORY] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Add ARD callbacks if enabled
    if enable_callbacks:
        device = next(model.parameters()).device
        
        callbacks = create_ard_callbacks(
            device=device,
            output_dir=output_dir,
            train_ds=train_dataset,
            val_ds=eval_dataset,
            ard_prior_samples=cfg.get("ard_prior_samples", 1000),
            batch_size=cfg.get("batch_size", 4),
            tokenizer=tokenizer,
            enable_plotting=cfg.get("enable_plotting", True),
            enable_resampling=cfg.get("enable_resampling", False),  # Disabled by default for CLM
            plot_start_epoch=cfg.get("plot_start_epoch", 2),
            plot_interval=cfg.get("plot_interval", 2)
        )
        
        # Add callbacks to trainer
        for callback in callbacks:
            trainer.add_callback(callback)
            
        print(f"[INFO] Added {len(callbacks)} ARD callbacks to trainer")
        for callback in callbacks:
            print(f"   - {callback.__class__.__name__}")
    
    return trainer

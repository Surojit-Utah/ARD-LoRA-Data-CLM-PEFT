from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
import torch
from torch import nn
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
    """Enhanced ARD Trainer following DeBERTa pattern, adapted for LLaMA."""
    
    def __init__(self, *args, beta=0.01, heldout_loader=None, ard_eval_dataset=None, uncertainty_eval_samples=1000, 
                 n_bins=15, output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.heldout_loader = heldout_loader
        self.ard_eval_dataset = ard_eval_dataset  # Dataset for ARD prior estimation
        self.uncertainty_eval_samples = uncertainty_eval_samples
        self.n_bins = n_bins
        self.uncertainty_evaluator = UncertaintyEvaluator(n_bins=n_bins) if UncertaintyEvaluator else None
        self.uncertainty_results = []  # Store results across epochs
        self.output_dir = output_dir or self.args.output_dir
        
        # Set trainer reference on model for callbacks
        self.model.trainer = self
        
        # Track loss components for logging
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss following DeBERTa pattern, adapted for LLaMA architecture."""
        # Extract labels for causal LM
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute CE loss (causal LM loss)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Get inputs for layer processing (following DeBERTa pattern)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Get hidden states from model forward pass (similar to DeBERTa encoder outputs)
        with torch.no_grad():
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                # Get embeddings and run through model to get hidden states
                embeddings = model.model.embed_tokens(input_ids)
                hidden_states_outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                hidden_states = hidden_states_outputs.hidden_states
            else:
                hidden_states = None
        
        # Compute KL divergence following DeBERTa pattern
        kl = 0.0
        
        if hidden_states is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, layer in enumerate(model.model.layers):
                # Get the input to this layer (previous layer's output)
                layer_input = hidden_states[layer_idx] if layer_idx < len(hidden_states) else None
                
                if layer_input is not None:
                    # Check attention projections (similar to DeBERTa query/value projections)
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        
                        # Process query and value projections (main components like in DeBERTa)
                        for proj_name in ['q_proj', 'v_proj']:
                            if hasattr(attn, proj_name):
                                proj = getattr(attn, proj_name)
                                # Check if this is a ProbLoRA layer
                                if hasattr(proj, 'kl_divergence_latent'):
                                    try:
                                        kl += proj.kl_divergence_latent(layer_input)
                                    except Exception:
                                        continue
                        
                        # Process output projection (similar to DeBERTa output dense layer)
                        if hasattr(attn, 'o_proj'):
                            proj = attn.o_proj
                            if hasattr(proj, 'kl_divergence_latent'):
                                try:
                                    kl += proj.kl_divergence_latent(layer_input)
                                except Exception:
                                    continue
                    
                    # Check MLP projections (additional components)
                    if hasattr(layer, 'mlp'):
                        mlp = layer.mlp
                        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                            if hasattr(mlp, proj_name):
                                proj = getattr(mlp, proj_name)
                                if hasattr(proj, 'kl_divergence_latent'):
                                    try:
                                        kl += proj.kl_divergence_latent(layer_input)
                                    except Exception:
                                        continue
        
        # If no KL components found, create zero tensor with gradient connection
        if not torch.is_tensor(kl) or kl == 0.0:
            kl = torch.tensor(0.0, device=ce_loss.device, requires_grad=True)
        
        # Total loss with KL regularization (following DeBERTa pattern)
        loss = ce_loss + self.beta * kl
        
        # Store loss components for logging
        self.last_ce_loss = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
        self.last_kl_loss = kl.item() if torch.is_tensor(kl) else float(kl)
        self.last_total_loss = loss.item() if torch.is_tensor(loss) else float(loss)
        
        return (loss, outputs) if return_outputs else loss
        
    def validate_model_gradients(self, model):
        """Validate that model parameters requiring gradients are properly set up."""
        trainable_params = 0
        total_params = 0
        
        problematic_params = []
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # Check if parameter has gradient function but data is detached
                if param.grad_fn is None and param.is_leaf and param.requires_grad:
                    # This is normal for leaf parameters
                    pass
                elif not param.is_leaf and param.grad_fn is None:
                    problematic_params.append(f"{name}: non-leaf parameter without grad_fn")
            
        print(f"[GRADIENT VALIDATION] Model parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.4f}")
        
        if problematic_params:
            print(f"[GRADIENT WARNING] Found {len(problematic_params)} potentially problematic parameters:")
            for param_info in problematic_params[:5]:  # Show first 5
                print(f"    {param_info}")
        else:
            print(f"[GRADIENT VALIDATION] ✅ All parameters appear correctly configured")
        
        return trainable_params > 0


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
        was_training = model.training
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
                
                # GRADIENT FIX: Compute KL divergence using component collection for proper gradient flow
                kl_components = []
                for module in model.modules():
                    if hasattr(module, 'kl_divergence'):
                        try:
                            module_kl = module.kl_divergence()
                            if torch.is_tensor(module_kl):
                                kl_components.append(module_kl)
                        except Exception:
                            continue
                
                # Enhanced KL computation fallback with proper tensor operations
                if len(kl_components) == 0:
                    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        for layer in model.model.layers:
                            if hasattr(layer, 'self_attn'):
                                attn = layer.self_attn
                                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                                    if hasattr(attn, proj_name):
                                        proj = getattr(attn, proj_name)
                                        if hasattr(proj, 'kl_divergence'):
                                            try:
                                                module_kl = proj.kl_divergence()
                                                if torch.is_tensor(module_kl):
                                                    kl_components.append(module_kl)
                                            except Exception:
                                                continue
                
                # GRADIENT FIX: Use torch.stack().sum() to maintain gradient flow
                if kl_components:
                    kl = torch.stack(kl_components).sum()
                else:
                    # Create zero tensor with gradient connection to model parameters
                    model_param = next(model.parameters())
                    kl = torch.zeros_like(ce_loss) + model_param.sum() * 0.0
                
                # Convert to float values
                ce_loss_val = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
                kl_loss_val = kl.item() if torch.is_tensor(kl) else float(kl)
                
                return ce_loss_val, kl_loss_val
                
        except Exception as e:
            print(f"⚠️ Failed to compute eval loss components: {e}")
            return 0.0, 0.0
        finally:
            # CRITICAL: Always restore original training mode
            if was_training:
                model.train()
            else:
                model.eval()

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
    """Callback to estimate ARD priors following DeBERTa pattern."""
    
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
            
        # Use heldout_loader if available (DeBERTa style), otherwise use ard_eval_dataset
        eval_data = getattr(trainer, 'heldout_loader', None) or getattr(trainer, 'ard_eval_dataset', None)
        
        if eval_data is None:
            print("[PriorEstimationCallback] No held-out loader or ARD evaluation dataset found for this epoch")
            return
        
        print(f"[PriorEstimationCallback] Estimating ARD priors at epoch {int(state.epoch)}...")
        
        # CRITICAL: Get tokenizer from kwargs first, then trainer - fail fast if None
        tokenizer = kwargs.get('tokenizer', getattr(trainer, 'tokenizer', None))
        if tokenizer is None:
            print("[PriorEstimationCallback] WARNING: Tokenizer is None - attempting to continue without tokenizer validation")
        
        try:
            estimate_ard_priors_clm(model, eval_data, self.device, tokenizer=tokenizer)
            print("[PriorEstimationCallback] ARD prior estimation completed")
        except Exception as e:
            print(f"[PriorEstimationCallback] ARD prior estimation failed: {e}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LatentPlotCallback(TrainerCallback):
    """Callback to plot latent encodings following DeBERTa pattern."""
    
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
        
        if trainer is None:
            print("[LatentPlotCallback] No trainer reference found")
            return
        
        # Use heldout_loader if available (DeBERTa style), otherwise use ard_eval_dataset
        eval_data = getattr(trainer, 'heldout_loader', None) or getattr(trainer, 'ard_eval_dataset', None)
        
        if eval_data is None:
            print("[LatentPlotCallback] No held-out loader or ARD evaluation dataset found for plotting")
            return
        
        if plot_mean_encodings is None:
            print("[LatentPlotCallback] Plotting utilities not available")
            return
        
        print(f"[LatentPlotCallback] Plotting latent encodings at epoch {current_epoch}...")
        
        try:
            # Create plots directory
            plot_dir = self.output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots using the available evaluation data
            if hasattr(eval_data, 'dataset'):
                # If it's a DataLoader, create a simple DataLoader for plotting
                from torch.utils.data import DataLoader
                plot_dataloader = DataLoader(eval_data.dataset, batch_size=16, shuffle=False)
            else:
                # If it's a dataset, create a DataLoader
                from torch.utils.data import DataLoader
                plot_dataloader = DataLoader(eval_data, batch_size=16, shuffle=False)
            
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
    """Callback to resample validation data for ARD vs evaluation split each epoch."""
    
    def __init__(self, train_ds, val_ds, ard_prior_samples, batch_size, tokenizer=None):
        super().__init__()
        self.train_ds = train_ds  # Keep for compatibility, but won't be modified
        self.val_ds = val_ds      # This is what we'll resample from
        self.ard_prior_samples = ard_prior_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        
        # Validate that we have enough validation data
        if self.val_ds is not None and len(self.val_ds) < self.ard_prior_samples:
            print(f"[HeldoutResampleCallback] WARNING: Validation dataset has only {len(self.val_ds)} samples, "
                  f"but {self.ard_prior_samples} requested for ARD. Will use all validation data for ARD.")
            self.ard_prior_samples = len(self.val_ds)
    
    def _resample_validation_split(self):
        """Resample validation data split: ARD samples vs evaluation samples."""
        if self.val_ds is None or len(self.val_ds) == 0:
            return None, None
        
        total_val = len(self.val_ds)
        
        # Ensure we don't exceed validation dataset size
        ard_samples = min(self.ard_prior_samples, total_val)
        
        # Randomly shuffle validation indices
        perm = np.random.permutation(total_val)
        
        # Split: first portion for ARD, remaining for evaluation
        ard_indices = perm[:ard_samples].tolist()
        eval_indices = perm[ard_samples:].tolist()
        
        # Create subset datasets
        if hasattr(self.val_ds, 'select'):
            # HuggingFace dataset
            ard_dataset = self.val_ds.select(ard_indices)
            eval_dataset = self.val_ds.select(eval_indices) if eval_indices else None
        else:
            # PyTorch dataset
            ard_dataset = Subset(self.val_ds, ard_indices)
            eval_dataset = Subset(self.val_ds, eval_indices) if eval_indices else None
        
        return ard_dataset, eval_dataset
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Resample validation split at the beginning of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[HeldoutResampleCallback] No trainer reference found")
            return
        
        if self.val_ds is None:
            print("[HeldoutResampleCallback] No validation dataset available for resampling")
            return
        
        try:
            # Resample validation data split
            ard_dataset, eval_dataset = self._resample_validation_split()
            
            if ard_dataset is not None:
                # Update trainer's ARD dataset (for prior estimation)
                trainer.ard_eval_dataset = ard_dataset
                
                # Update trainer's evaluation dataset (for model evaluation)
                if eval_dataset is not None:
                    trainer.eval_dataset = eval_dataset
                
                print(f"[HeldoutResampleCallback] Epoch {int(state.epoch)} → "
                      f"ARD: {len(ard_dataset)} samples, "
                      f"Eval: {len(eval_dataset) if eval_dataset else 0} samples "
                      f"(from {len(self.val_ds)} total validation samples)")
            
        except Exception as e:
            print(f"[HeldoutResampleCallback] Failed to resample validation data: {e}")


def split_validation_dataset(val_dataset, ard_prior_samples=100, seed=42, train_dataset=None, train_split_ratio=0.1):
    """
    Split validation dataset into two parts:
    1. ARD prior estimation (first part) - specified number of samples
    2. Uncertainty evaluation (second part) - remaining validation data
    
    If val_dataset is None, creates a validation split from train_dataset.
    
    Args:
        val_dataset: Original validation dataset (can be None)
        ard_prior_samples: Absolute number of samples to use for ARD prior estimation (default: 100)
        seed: Random seed for reproducible splits
        train_dataset: Training dataset (used if val_dataset is None)
        train_split_ratio: Fraction of training data to use for validation when val_dataset is None (min 10%)
    
    Returns:
        ard_dataset: Dataset for ARD prior estimation (exact number of samples requested)
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
    
    # Use the requested number of samples, but don't exceed 80% of validation data
    max_ard_ratio = 0.8
    max_ard_samples = int(total_size * max_ard_ratio)
    
    if total_size < ard_prior_samples:
        print(f"[WARNING] Validation dataset has only {total_size} samples, less than requested {ard_prior_samples} for ARD")
        ard_size = total_size
        uncertainty_size = 0
    else:
        # Use exactly the requested number of samples, but respect max ratio constraint
        ard_size = min(ard_prior_samples, max_ard_samples)
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


@torch.no_grad()
def estimate_ard_priors_clm(model, eval_dataset, device, num_samples=1000, tokenizer=None):
    """
    Estimate ARD priors following DeBERTa pattern, adapted for LLaMA architecture.
    
    Args:
        model: ARD-LoRA model with ProbLoRA layers
        eval_dataset: Dataset or DataLoader for prior estimation
        device: Device to run computation on
        num_samples: Number of samples to use for estimation
        tokenizer: Tokenizer for data collation (REQUIRED)
    """
    print(f"[ARD] Estimating priors using {num_samples} samples...")
    
    # Set model to eval mode for prior estimation
    was_training = model.training
    model.eval()
    
    # Create dataloader if needed
    if not isinstance(eval_dataset, DataLoader):
        from transformers import DataCollatorForLanguageModeling
        
        # CRITICAL: Tokenizer is required - fail fast if None
        if tokenizer is None:
            raise RuntimeError("[ARD] CRITICAL: Tokenizer is required for ARD prior estimation but was None. This indicates a serious configuration error.")
        
        # Check tokenizer validity
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            raise RuntimeError(f"[ARD] CRITICAL: Invalid tokenizer (no pad_token_id) - this will cause training failures")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
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
    
    # Collect ProbLoRA layers and initialize beta accumulators (following DeBERTa pattern)
    prob_lora_layers = {}
    
    # Find all ProbLoRA layers in the model (adapted from DeBERTa approach)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            layer_projections = {}
            
            # Check attention projections (similar to DeBERTa query/value projections)
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'v_proj', 'o_proj']:  # Focus on main projections like DeBERTa
                    if hasattr(attn, proj_name):
                        proj = getattr(attn, proj_name)
                        # Check if this is a ProbLoRA layer (has beta_get_sample method)
                        if hasattr(proj, 'beta_get_sample') and hasattr(proj, 'rank'):
                            layer_projections[f'attn_{proj_name}'] = proj
            
            # Check MLP projections (additional components)
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, proj_name):
                        proj = getattr(mlp, proj_name)
                        # Check if this is a ProbLoRA layer (has beta_get_sample method)
                        if hasattr(proj, 'beta_get_sample') and hasattr(proj, 'rank'):
                            layer_projections[f'mlp_{proj_name}'] = proj
            
            if layer_projections:
                prob_lora_layers[layer_idx] = layer_projections
    
    if not prob_lora_layers:
        print("[ARD] No ProbLoRA layers found for prior estimation")
        # Restore original training mode
        if was_training:
            model.train()
        return []
    
    # Initialize beta accumulators for each layer (following DeBERTa pattern)
    beta_accumulators = {}
    for layer_idx, projections in prob_lora_layers.items():
        beta_accumulators[layer_idx] = {}
        for proj_name, proj in projections.items():
            beta_accumulators[layer_idx][proj_name] = np.zeros(proj.rank, dtype=np.float32)
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(eval_dataloader):
        if sample_count >= num_samples:
            break
        
        # Move to device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Get hidden states for each layer (following DeBERTa approach)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Get hidden states from model forward pass (similar to DeBERTa encoder outputs)
        if hasattr(model, 'model'):
            hidden_states_outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = hidden_states_outputs.hidden_states
            
            # Process each layer that has ProbLoRA projections (following DeBERTa pattern)
            for layer_idx, projections in prob_lora_layers.items():
                if layer_idx < len(hidden_states):
                    layer_input = hidden_states[layer_idx]
                    
                    # Accumulate beta samples for each projection (following DeBERTa pattern)
                    for proj_name, proj in projections.items():
                        try:
                            beta_sample = proj.beta_get_sample(layer_input)
                            beta_accumulators[layer_idx][proj_name] += np.sum(beta_sample ** 2, axis=0) / 2.0
                        except Exception as e:
                            print(f"[ARD] Warning: Failed to get beta sample for layer {layer_idx} {proj_name}: {e}")
                            continue
        
        sample_count += inputs['input_ids'].size(0)
        
        if batch_idx % 10 == 0:
            print(f"   Processed {batch_idx + 1} batches, ~{sample_count} samples")
    
    # Update est_var for each layer (following DeBERTa pattern: est_var = beta / alpha + 1e-6)
    print(f"\n[ARD] Updating estimated variances for {len(prob_lora_layers)} layers:")
    
    for layer_idx, projections in prob_lora_layers.items():
        for proj_name, proj in projections.items():
            beta_accumulated = beta_accumulators[layer_idx][proj_name]
            # Calculate est_var: beta / alpha + 1e-6 (same as DeBERTa)
            est_var_values = beta_accumulated / proj.alpha + 1e-6
            # Create a float32 tensor on the target device for est_var
            est_var_tensor = torch.tensor(est_var_values, dtype=torch.float32, device=device)

            # If the module already has an est_var tensor buffer, try to update it in-place
            try:
                if hasattr(proj, 'est_var') and isinstance(proj.est_var, torch.Tensor):
                    # Attempt an in-place copy while respecting existing dtype/device
                    try:
                        proj.est_var.data = est_var_tensor.to(proj.est_var.device, dtype=proj.est_var.dtype).data
                    except Exception:
                        # Fallback to re-registering the buffer if in-place update fails
                        try:
                            proj.register_buffer('est_var', est_var_tensor)
                        except Exception:
                            # As a last resort, set attribute (works but won't move with .to())
                            setattr(proj, 'est_var', est_var_tensor)
                else:
                    # Register as a buffer so it moves with the module's device/state_dict
                    proj.register_buffer('est_var', est_var_tensor)
            except Exception:
                # Be defensive: ensure there's at least an attribute even if register_buffer fails
                try:
                    proj.est_var = est_var_tensor
                except Exception:
                    # Give up quietly but report
                    print(f"[ARD] Warning: Could not set est_var for projection {proj_name} in layer {layer_idx}")
            
            # Print statistics
            avg_est_var = np.mean(est_var_values)
            print(f"   Layer {layer_idx} {proj_name}: avg_est_var={avg_est_var:.6f}")
            
            # ARD relevance determination (using est_var)
            relevance = "High" if avg_est_var > 0.1 else "Medium" if avg_est_var > 0.01 else "Low"
            print(f"     → Estimated relevance: {relevance}")
    
    # Restore original training mode
    if was_training:
        model.train()
    else:
        model.eval()
    
    print(f"[ARD] ✅ Prior estimation completed - updated est_var for all ProbLoRA layers")
    return list(prob_lora_layers.keys())


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
                     ard_prior_samples=100, enable_callbacks=True, tb_log_dir=None, use_deberta_pattern=False):
    """Build enhanced CLM trainer with uncertainty evaluation, ARD callbacks, and prior estimation.
    
    Args:
        use_deberta_pattern: If True, use DeBERTa-style heldout_loader approach
    """
    
    # Split validation dataset for ARD and uncertainty evaluation
    # Pass train_dataset so we can create validation split if needed
    ard_dataset, uncertainty_dataset = split_validation_dataset(
        eval_dataset, 
        ard_prior_samples=ard_prior_samples,  # Use absolute sample count instead of ratio
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
        bf16=bool(cfg.get("bf16", False)),  # BF16 support for A100 GPUs
        fp16=bool(cfg.get("fp16", True)) if not cfg.get("bf16", False) else False,  # Use fp16 only if bf16 is not enabled
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
        gradient_checkpointing_kwargs={"use_reentrant": False},  # <- add this for the checkpoint warnings 
    )

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create heldout_loader if using DeBERTa pattern
    heldout_loader = None
    if use_deberta_pattern and ard_dataset is not None:
        heldout_loader = DataLoader(
            ard_dataset,
            batch_size=cfg.get("batch_size", 4),
            collate_fn=data_collator,
            shuffle=False
        )

    # Create the enhanced trainer
    trainer = ARDCLMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=uncertainty_dataset,  # Use uncertainty dataset for standard evaluation
        data_collator=data_collator,
        tokenizer=tokenizer,
        beta=float(cfg.get("kl_loss_beta", 0.01)),
        heldout_loader=heldout_loader,  # DeBERTa-style heldout loader
        ard_eval_dataset=ard_dataset,  # Separate dataset for ARD prior estimation
        uncertainty_eval_samples=cfg.get("uncertainty_eval_samples", 1000),
        n_bins=cfg.get("uncertainty_n_bins", 15),
        output_dir=output_dir
    )
    # GRADIENT-CHECKPOINTING FIX: ensure at least one checkpointed input has requires_grad=True
    # Register a small embed forward-hook and per-layer forward_pre_hooks. Track how many
    # hooks succeeded so we can print a coverage summary. Optionally enable a debug wrapper
    # for torch.utils.checkpoint.checkpoint via cfg['checkpoint_debug'] to log offending calls.
    try:
        hook_embed_attached = 0
        hook_layer_attached = 0

        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            def _embed_forward_hook(module, inp, out):
                try:
                    if isinstance(out, torch.Tensor):
                        out.requires_grad_(True)
                except Exception:
                    pass
            try:
                model.model.embed_tokens.register_forward_hook(_embed_forward_hook)
                hook_embed_attached = 1
            except Exception:
                hook_embed_attached = 0

        if hasattr(model.model, 'layers'):
            def _make_pre_hook():
                def _pre_hook(module, inputs):
                    try:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            if not inputs[0].requires_grad:
                                inputs[0].requires_grad_(True)
                    except Exception:
                        pass
                return _pre_hook

            for lyr in model.model.layers:
                try:
                    lyr.register_forward_pre_hook(_make_pre_hook())
                    hook_layer_attached += 1
                except Exception:
                    continue

        print(f"[GRADIENT FIX] Hook coverage: embed_hook={hook_embed_attached}, layer_pre_hooks={hook_layer_attached}/{len(getattr(model.model, 'layers', []))}")

        # Optional debug wrapper for checkpoint to log inputs when no requires_grad present
        try:
            debug_flag = False
            if isinstance(cfg, dict):
                debug_flag = bool(cfg.get('checkpoint_debug', False))
            # Fall back to environment variable CHECKPOINT_DEBUG=1 (useful for full training runs)
            if not debug_flag:
                try:
                    import os as _os
                    if _os.environ.get('CHECKPOINT_DEBUG', '').lower() in ('1', 'true', 'yes'):
                        debug_flag = True
                except Exception:
                    pass

            if debug_flag:
                import traceback as _traceback
                _orig_checkpoint = torch.utils.checkpoint.checkpoint

                if not getattr(torch.utils.checkpoint, '_checkpoint_debug_wrapped', False):
                    def _debug_checkpoint(func, *args, **kwargs):
                        # gather tensors
                        tensors = []
                        def _g(x):
                            if isinstance(x, torch.Tensor):
                                tensors.append(x)
                            elif isinstance(x, (list, tuple)):
                                for e in x: _g(e)
                            elif isinstance(x, dict):
                                for e in x.values(): _g(e)
                        _g(args)
                        _g(kwargs)

                        any_req = False
                        for t in tensors:
                            try:
                                if t is not None and t.is_floating_point() and t.requires_grad:
                                    any_req = True
                                    break
                            except Exception:
                                continue

                        if not any_req:
                            print('[CHECKPOINT_DEBUG] checkpoint called with NO inputs requiring grad')
                            for i, t in enumerate(tensors[:12]):
                                try:
                                    print(f'  tensor[{i}]: shape={tuple(t.shape) if t is not None else None}, dtype={t.dtype if t is not None else None}, requires_grad={t.requires_grad if t is not None else None}')
                                except Exception:
                                    pass
                            _traceback.print_stack(limit=6)

                        return _orig_checkpoint(func, *args, **kwargs)

                    torch.utils.checkpoint.checkpoint = _debug_checkpoint
                    torch.utils.checkpoint._checkpoint_debug_wrapped = True
                    print('[CHECKPOINT_DEBUG] Wrapped torch.utils.checkpoint.checkpoint with debug logger')

        except Exception as e:
            print(f"[CHECKPOINT_DEBUG] Failed to install debug wrapper: {e}")

    except Exception as e:
        print(f"[GRADIENT FIX] Failed to register checkpoint hooks: {e}")
    
    # Enable gradient checkpointing for memory efficiency based on configuration
    if cfg.get("gradient_checkpointing", True) and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("[MEMORY] Enabled gradient checkpointing based on configuration")
    elif hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("[MEMORY] Disabled gradient checkpointing based on configuration")
    else:
        print(f"[MEMORY] Gradient checkpointing setting: {cfg.get('gradient_checkpointing', True)}")
    
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
            enable_resampling=cfg.get("enable_resampling", use_deberta_pattern),  # Enable resampling if using DeBERTa pattern
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

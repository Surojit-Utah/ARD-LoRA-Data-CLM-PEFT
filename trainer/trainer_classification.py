
"""
ARD-LoRA Classification Trainer for Multiple Choice QA
======================================================

This module provides a specialized trainer for classification tasks like ARC-Easy,
where we predict a single answer from K choices using last-token logits.

Key Differences from Standard CLM:
1. Extract ONLY last token logits (not full sequence)
2. Filter logits to K answer tokens (e.g., A, B, C, D, E)
3. Compute CE loss over K classes (not full vocabulary)
4. KL divergence computed same way (over hidden states)
"""

import torch
from torch import nn
from typing import Dict, Any, Optional
from transformers import Trainer
from evaluate.uncertainty_metrics import UncertaintyEvaluator
import numpy as np


class ARDClassificationTrainer(Trainer):
    """
    Enhanced ARD Trainer for classification tasks with last-token prediction.
    
    Adapted from ARDCLMTrainer but with classification-specific loss computation.
    """
    
    def __init__(
        self,
        *args,
        config=None,  # NEW: Accept config dict
        beta=None,
        ard_heldout_loader=None,
        n_bins=None,
        output_dir=None,
        ard_prior_samples=None,
        target_attention_layers=None,
        target_ids=None,  # NEW: Token IDs for valid answers
        num_classes=None,    # NEW: Number of classes (K)
        verbose=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Instance variable to control KL loss computation
        self.use_kl = False  # Set to True to enable KL divergence loss
        
        # Load parameters from config with fallbacks
        if config is not None:
            self.beta = beta if beta is not None else config.get('kl_loss_beta')
            self.n_bins = n_bins if n_bins is not None else config.get('uncertainty_n_bins')
            self.ard_prior_samples = ard_prior_samples if ard_prior_samples is not None else config.get('ard_prior_samples')
            self.num_classes = num_classes if num_classes is not None else config.get('num_classes')
            self.verbose = verbose if verbose is not None else config.get('verbose')
            
            # Target attention layers
            if target_attention_layers is None:
                target_attention_layers = config.get('target_attention_layers')
        else:
            self.beta = beta if beta is not None else 0.01
            self.n_bins = n_bins if n_bins is not None else 15
            self.ard_prior_samples = ard_prior_samples if ard_prior_samples is not None else 100
            self.num_classes = num_classes if num_classes is not None else 4  # Default to 4 (A, B, C, D)
            self.verbose = verbose
            
            if target_attention_layers is None:
                raise ValueError("target_attention_layers must be provided")
        
        self.ard_heldout_loader = ard_heldout_loader
        self.uncertainty_evaluator = UncertaintyEvaluator(n_bins=self.n_bins)
        self.uncertainty_results = []
        self.output_dir = output_dir or self.args.output_dir
        
        # Classification-specific parameters
        if target_ids is None:
            raise ValueError("target_ids must be provided for classification mode")
        self.target_ids = target_ids  # Tensor of shape [num_classes]
        
        # Store layer configuration
        self.target_attention_layers = target_attention_layers
        
        # Set trainer reference on model for callbacks
        self.model.trainer = self
        
        # Track loss components for logging
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        
        print(f"[CLASSIFICATION] ARDClassificationTrainer initialized:")
        print(f"[CLASSIFICATION]   Num classes: {self.num_classes}")
        print(f"[CLASSIFICATION]   Target IDs: {self.target_ids.tolist()}")
        print(f"[CLASSIFICATION]   KL beta: {self.beta}")
        print(f"[CLASSIFICATION]   Use KL loss: {self.use_kl}")
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        """
        Compute classification loss with last-token prediction.
        
        Key Steps:
        1. Forward pass to get full logits [batch, seq_len, vocab_size]
        2. Extract ONLY last token logits [batch, vocab_size]
        3. Filter to K answer tokens [batch, num_classes]
        4. Compute CE loss over K classes
        5. Compute KL divergence over hidden states (same as CLM)
        """
        # Extract gold classes (class indices)
        # S2ClassDataset uses 'labels', our custom datasets might use 'classes'
        # Support both for compatibility
        if "classes" in inputs:
            classes = inputs.pop("classes")
        elif "labels" in inputs:
            classes = inputs.pop("labels")
        else:
            raise ValueError("Classification trainer requires 'classes' or 'labels' in inputs")
        
        # CRITICAL: Verify model is in training mode
        if not hasattr(self, '_training_mode_checked'):
            self._training_mode_checked = True
            print(f"\n[CRITICAL] Model training mode in compute_loss: {model.training}")
            # Check if any parameters require gradients
            trainable = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"[CRITICAL] Trainable parameters in compute_loss: {trainable}")
        
        # Forward pass with hidden states for both CE loss and KL computation
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        hidden_states = outputs.hidden_states
        
        # ===== CLASSIFICATION-SPECIFIC LOGIC =====
        
        # Step 1: Extract ONLY the last token logits
        # CRITICAL FIX: Use attention mask to find last non-pad token per example
        attn = inputs["attention_mask"]  # [batch_size, seq_len]
        last_idx = attn.long().sum(dim=1) - 1  # [batch_size]
        batch_indices = torch.arange(logits.size(0), device=logits.device)  # [batch_size]
        last_token_logits = logits[batch_indices, last_idx, :]  # [batch_size, vocab_size]
        
        # Step 2: Filter to ONLY valid answer tokens
        # target_ids shape: [num_classes] (e.g., [319, 350, 315, 360, 382] for A,B,C,D,E)
        target_ids_device = self.target_ids.to(last_token_logits.device)
        filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
        # Shape: [batch_size, num_classes]
        
        # Step 3: Compute cross-entropy loss over K classes
        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(filtered_logits, classes)
        
        # Debug info (print once per epoch)
        if not hasattr(self, '_debug_classification_printed'):
            self._debug_classification_printed = True
            print(f"\n[CLASSIFICATION LOSS DEBUG]:")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Last token logits shape: {last_token_logits.shape}")
            print(f"  Filtered logits shape: {filtered_logits.shape}")
            print(f"  Classes shape: {classes.shape}")
            print(f"  Classes: {classes.tolist()}")
            print(f"  CE Loss: {ce_loss.item():.4f}")
        
        # ===== KL DIVERGENCE COMPUTATION (SAME AS CLM) =====
        
        kl = 0.0
        
        if self.use_kl:
            # KL computation only when enabled
            kl_debug_info = {}
            total_kl_layers = 0
            
            # Track current epoch for debug printing
            current_epoch = int(getattr(self.state, 'epoch', 0)) if hasattr(self, 'state') else 0
            
            # Initialize or check if we need to print debug for this epoch
            if not hasattr(self, '_last_gradient_debug_epoch'):
                self._last_gradient_debug_epoch = -1
            
            # Only print once per epoch when epoch changes
            if current_epoch > self._last_gradient_debug_epoch:
                self._last_gradient_debug_epoch = current_epoch
                if hidden_states is not None:
                    print(f"\n[GRADIENT DEBUG] Epoch {current_epoch} - Hidden States Analysis:")
                    print(f"[GRADIENT DEBUG]   Number of hidden state layers: {len(hidden_states)}")
                    print(f"[GRADIENT DEBUG]   Hidden states[0] requires_grad: {hidden_states[0].requires_grad}")
                    print(f"[GRADIENT DEBUG]   Hidden states[0] has grad_fn: {hidden_states[0].grad_fn is not None}")
            
            # Compute KL divergence over ProbLoRA layers
            if hidden_states is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for layer_idx, layer in enumerate(model.model.layers):
                    layer_input = hidden_states[layer_idx] if layer_idx < len(hidden_states) else None
                    
                    if layer_input is not None:
                        if hasattr(layer, 'self_attn') and self.target_attention_layers:
                            attn = layer.self_attn
                            layer_kl_total = 0.0
                            layer_proj_count = 0
                            
                            for proj_name in self.target_attention_layers:
                                if hasattr(attn, proj_name):
                                    proj = getattr(attn, proj_name)
                                    if hasattr(proj, 'kl_divergence_latent'):
                                        try:
                                            proj_kl = proj.kl_divergence_latent(layer_input)
                                            kl += proj_kl
                                            layer_kl_total += proj_kl.item() if torch.is_tensor(proj_kl) else float(proj_kl)
                                            layer_proj_count += 1
                                        except Exception:
                                            continue
                            
                            if layer_proj_count > 0:
                                kl_debug_info[f"layer_{layer_idx}"] = {
                                    "projections_processed": layer_proj_count,
                                    "target_projections": list(self.target_attention_layers),
                                    "layer_kl_total": layer_kl_total
                                }
                                total_kl_layers += 1
            
            # If no KL components found, create zero tensor with gradient connection
            # Safe check to avoid "Boolean value of Tensor is ambiguous" error
            if not torch.is_tensor(kl) or (torch.is_tensor(kl) and float(kl.detach().item()) == 0.0):
                kl = torch.tensor(0.0, device=ce_loss.device, requires_grad=True)
        
        # Combine losses
        if self.use_kl:
            total_loss = ce_loss + self.beta * kl
        else:
            total_loss = ce_loss  # Keep tensor, don't add scalar 0.0
        
        # Store for logging
        self.last_ce_loss = ce_loss.item()
        self.last_kl_loss = kl.item() if torch.is_tensor(kl) else float(kl) if self.use_kl else 0.0
        self.last_total_loss = total_loss.item()
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add gradient sanity check after first backward.
        
        Args:
            model: The model being trained
            inputs: The inputs and targets for the model
            num_items_in_batch: Number of items in the batch (for gradient accumulation)
        """
        # Call parent training_step (this does forward, backward, optimizer step)
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Gradient sanity check: run once after first backward pass
        if not hasattr(self, '_gradient_sanity_checked'):
            self._gradient_sanity_checked = True
            print("\n" + "="*60)
            print("[GRAD SANITY CHECK] After first backward pass")
            print("="*60)
            
            nz, tot = 0, 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    tot += 1
                    if p.grad is not None and torch.count_nonzero(p.grad).item() > 0:
                        nz += 1
            
            print(f"[GRAD] Trainable parameters with nonzero gradients: {nz}/{tot}")
            
            if nz == 0:
                print("[GRAD] ⚠️  WARNING: NO gradients computed! Check loss computation.")
            elif nz < tot:
                print(f"[GRAD] ⚠️  WARNING: Only {nz}/{tot} parameters have gradients!")
            else:
                print(f"[GRAD] ✅ All trainable parameters have gradients.")
            
            # Optional: Show a few gradient samples
            print(f"[GRAD] Sample gradients (first 3 LoRA parameters):")
            count = 0
            for n, p in model.named_parameters():
                if p.requires_grad and 'lora' in n.lower() and p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    print(f"[GRAD]   {n}: grad_norm={grad_norm:.6f}")
                    count += 1
                    if count >= 3:
                        break
            print("="*60 + "\n")
        
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for classification tasks.
        
        Returns predictions as class probabilities.
        """
        # Extract classes
        classes = inputs.pop("classes") if "classes" in inputs else None
        inputs.pop("labels", None)
        
        has_labels = classes is not None
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
            logits = outputs.logits
            
            # Extract last token and filter to answer tokens
            # CRITICAL FIX: Use attention mask to find last non-pad token per example
            attn = inputs["attention_mask"]  # [batch_size, seq_len]
            last_idx = attn.long().sum(dim=1) - 1  # [batch_size]
            batch_indices = torch.arange(logits.size(0), device=logits.device)  # [batch_size]
            last_token_logits = logits[batch_indices, last_idx, :]  # [batch_size, vocab_size]
            target_ids_device = self.target_ids.to(last_token_logits.device)
            filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
            
            # Compute loss if we have labels
            if has_labels:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(filtered_logits, classes)
            else:
                loss = None
        
        # Convert logits to predictions (class indices)
        preds = torch.argmax(filtered_logits, dim=-1)
        
        # Return in format expected by Trainer
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, preds, classes)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Enhanced evaluation with classification metrics.
        """
        # Call parent evaluate
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Add classification-specific metrics if needed
        # (accuracy, F1, etc. can be added here)
        
        return metrics


def build_classification_trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
    config,  # NEW: Configuration dict
    ard_heldout_loader=None,
    target_ids=None,
    num_classes=None,
    **kwargs
):
    """
    Convenience function to build ARDClassificationTrainer with proper configuration.
    
    Args:
        model: The model to train
        args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator function
        tokenizer: Tokenizer
        config: Configuration dictionary (from YAML)
        ard_heldout_loader: DataLoader for ARD prior estimation
        target_ids: Tensor of target token IDs for answer classes
        num_classes: Number of classes (K)
        **kwargs: Additional trainer arguments
    
    Returns:
        ARDClassificationTrainer instance
    """
    # Get num_classes from config if not provided
    if num_classes is None:
        num_classes = config.get('num_classes')
    
    trainer = ARDClassificationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        config=config,  # Pass config
        ard_heldout_loader=ard_heldout_loader,
        target_ids=target_ids,
        num_classes=num_classes,
        **kwargs
    )
    
    return trainer
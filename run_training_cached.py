"""
ARD-LoRA Training with Bayesian-PEFT Datasets (Cached)
======================================================

This script demonstrates how to train ARD-LoRA using Bayesian-PEFT datasets
with local caching for persistent data storage.

Key Features:
1. Downloads datasets once and caches locally (e.g., Google Drive)
2. Leverages Bayesian-PEFT's proven dataset classes
3. Supports multiple dataset types (S2S, classification)
4. Compatible with ARD-LoRA probabilistic training
"""

import os
from pathlib import Path
from config import CONFIG
from model.model_llama import inject_problora_llama
from trainer.trainer_clm import ARDCLMTrainer, estimate_ard_priors_clm, build_clm_trainer, create_ard_callbacks
from dataloader.bayesian_peft_cached import load_bayesian_peft_with_caching
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.io import get_output_dirs, free_memory


def _merge_config(defaults: dict):
    """Merge configuration with hierarchy: defaults -> top-level -> model -> dataset"""
    cfg = CONFIG or {}
    merged = dict(defaults)
    
    # Apply top-level defaults
    merged.update(cfg.get("defaults", {}))
    
    # Apply model-specific defaults
    model_name = merged["model_name"]
    if "models" in cfg and model_name in cfg["models"]:
        model_cfg = cfg["models"][model_name]
        merged.update(model_cfg.get("defaults", {}))
    
    # Apply dataset-specific config
    dataset_name = merged["dataset_name"]
    if "datasets" in cfg and dataset_name in cfg["datasets"]:
        dataset_cfg = cfg["datasets"][dataset_name]
        merged.update(dataset_cfg)
    
    # Validate and fix data types for critical parameters
    _validate_config_types(merged)
    
    return merged


def _validate_config_types(config):
    """Validate and fix data types for critical configuration parameters"""
    # Ensure numeric parameters are correct types
    float_params = ["learning_rate", "kl_loss_beta", "warmup_ratio", "weight_decay", "scaling"]
    int_params = ["rank", "batch_size", "train_epochs", "max_len", "ard_prior_samples", 
                  "uncertainty_eval_samples", "uncertainty_n_bins", "gradient_accumulation_steps",
                  "runId", "num_labels", "max_validation_samples", "plot_start_epoch", "plot_interval"]
    bool_params = ["fp16", "bf16", "load_in_4bit", "use_cache", "gradient_checkpointing", 
                   "enable_callbacks", "enable_plotting", "enable_resampling", "use_google_drive"]
    
    # Convert string numbers to floats
    for param in float_params:
        if param in config and isinstance(config[param], str):
            try:
                config[param] = float(config[param])
                print(f"[CONFIG] Converted {param} from string to float: {config[param]}")
            except ValueError:
                print(f"[WARNING] Could not convert {param} to float: {config[param]}")
    
    # Convert string numbers to ints
    for param in int_params:
        if param in config and isinstance(config[param], str):
            try:
                config[param] = int(config[param])
                print(f"[CONFIG] Converted {param} from string to int: {config[param]}")
            except ValueError:
                print(f"[WARNING] Could not convert {param} to int: {config[param]}")
    
    # Validate CLM-specific settings
    if config["num_labels"] != 0:
        print(f"[WARNING] CLM training should have num_labels=0, but got {config['num_labels']}")
        config["num_labels"] = 0
        print(f"[CONFIG] Reset num_labels to 0 for CLM training")
    
    # Validate memory optimization settings for A100
    if not config["use_cache"]:
        print(f"[CONFIG] ✅ KV caching disabled for memory optimization")
    if config["gradient_checkpointing"]:
        print(f"[CONFIG] ✅ Gradient checkpointing enabled for memory optimization")
    if config["bf16"]:
        print(f"[CONFIG] ✅ BF16 precision enabled for A100 GPU optimization")


def _validate_tokenizer_alignment(tokenizer):
    """Validate tokenizer configuration for CLM training"""
    print(f"[TOKENIZER] Validation for CLM training:")
    print(f"[TOKENIZER]   Model: {tokenizer.name_or_path}")
    print(f"[TOKENIZER]   Vocab size: {tokenizer.vocab_size}")
    print(f"[TOKENIZER]   BOS token: {repr(tokenizer.bos_token)} (id: {tokenizer.bos_token_id})")
    print(f"[TOKENIZER]   EOS token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
    print(f"[TOKENIZER]   PAD token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
    print(f"[TOKENIZER]   UNK token: {repr(tokenizer.unk_token)} (id: {tokenizer.unk_token_id})")
    
    # Validate LLaMA-2 specific expectations
    if "llama" in tokenizer.name_or_path.lower():
        expected_eos_id = 2
        expected_bos_id = 1
        expected_unk_id = 0
        
        # Check EOS token (most critical for CLM)
        if tokenizer.eos_token_id == expected_eos_id:
            print(f"[TOKENIZER] ✅ EOS token ID correct: {tokenizer.eos_token_id}")
        else:
            print(f"[TOKENIZER] ⚠️  EOS token ID unexpected: got {tokenizer.eos_token_id}, expected {expected_eos_id}")
        
        # Check PAD token alignment
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print(f"[TOKENIZER] ✅ PAD token aligned with EOS: {tokenizer.pad_token_id}")
        else:
            print(f"[TOKENIZER] ⚠️  PAD token not aligned with EOS: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
        
        # Check BOS token
        if tokenizer.bos_token_id == expected_bos_id:
            print(f"[TOKENIZER] ✅ BOS token ID correct: {tokenizer.bos_token_id}")
        else:
            print(f"[TOKENIZER] ⚠️  BOS token ID unexpected: got {tokenizer.bos_token_id}, expected {expected_bos_id}")
    
    # Check for potential issues
    if tokenizer.pad_token_id is None:
        print(f"[TOKENIZER] ❌ ERROR: pad_token_id is None - this will cause training issues!")
        raise ValueError("pad_token_id cannot be None for CLM training")
    
    if tokenizer.pad_token_id == tokenizer.unk_token_id:
        print(f"[TOKENIZER] ⚠️  WARNING: pad_token_id equals unk_token_id - this may cause issues")
    
    print(f"[TOKENIZER] ✅ Tokenizer validation complete")


def setup_cache_directory(config):
    """
    Setup caching directory based on configuration.
    """
    # Check if Google Drive path is available (for Colab usage)
    drive_cache = "/content/drive/MyDrive/ARD_LoRA_Data_Cache"

    # Use Google Drive for persistent caching
    os.makedirs(drive_cache, exist_ok=True)
    print(f"[INFO] Using Google Drive cache: {drive_cache}")

    return drive_cache


def load_model_with_problora(config):
    """Load LLaMA2 model and inject ProbLoRA layers"""
    model_name_or_path = config["model_name_or_path"]
    tokenizer_name = config.get("tokenizer_name") or model_name_or_path
    
    print(f"[INFO] Loading model: {model_name_or_path}")
    
    # Model loading arguments
    model_kwargs = {}
    if config["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    # Set torch dtype based on precision preference (bf16 preferred on A100)
    import torch
    if config["bf16"]:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config["fp16"]:
        model_kwargs["torch_dtype"] = torch.float16
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Configure pad token for CLM training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[TOKENIZER] Set pad_token to eos_token: {repr(tokenizer.pad_token)}")
    
    # Validate tokenizer configuration for CLM
    _validate_tokenizer_alignment(tokenizer)
    
    # Configure use_cache for memory optimization
    original_use_cache = getattr(model.config, 'use_cache', None)
    print(f"[MEMORY] Original model.config.use_cache: {original_use_cache}")
    
    # Set use_cache based on configuration (disable for training to save memory)
    use_cache_setting = config["use_cache"]
    model.config.use_cache = use_cache_setting
    print(f"[MEMORY] Set model.config.use_cache to: {model.config.use_cache}")
    if not use_cache_setting:
        print(f"[MEMORY] KV caching disabled during training to reduce memory usage on 40GB A100")
    
    # Inject ProbLoRA with numerical stability parameters
    model = inject_problora_llama(
        model,
        rank=config["rank"],
        scaling=config["scaling"],
        num_tokens=config["max_len"],
        ard_prior_samples=config["ard_prior_samples"],
        logvar_clamp_min=config["logvar_clamp_min"],
        logvar_clamp_max=config["logvar_clamp_max"],
        beta_logvar_clamp_min=config["beta_logvar_clamp_min"],
        beta_logvar_clamp_max=config["beta_logvar_clamp_max"],
        sample_clamp_min=config["sample_clamp_min"],
        sample_clamp_max=config["sample_clamp_max"],
        attn_implementation=config["attn_implementation"],
        target_attention_layers=config["target_attention_layers"]  # Read from YAML config
    )
    
    # Freeze base parameters and unfreeze LoRA parameters
    trainable_count = 0
    lora_patterns = ['lora_a', 'lora_b', '.a.', '.b.', 'lora', 'adapter']
    
    print("[DEBUG] Analyzing parameter names for LoRA detection...")
    all_param_names = []
    quantized_params_skipped = 0
    
    for name, param in model.named_parameters():
        all_param_names.append(name)
        # More comprehensive LoRA parameter detection
        is_lora = any(pattern in name.lower() for pattern in lora_patterns)
        
        if is_lora:
            # CRITICAL: Debug parameter details before setting gradients
            print(f"[DEBUG] Found LoRA parameter: {name}")
            print(f"[DEBUG]   Shape: {param.shape}")
            print(f"[DEBUG]   Dtype: {param.dtype}")
            print(f"[DEBUG]   Is floating point: {param.dtype.is_floating_point}")
            
            # CRITICAL: Only set gradients on floating-point parameters
            if param.dtype.is_floating_point:
                param.requires_grad = True
                trainable_count += 1
                print(f"[DEBUG] ✅ Trainable LoRA param: {name} (shape: {param.shape}, dtype: {param.dtype})")
            else:
                quantized_params_skipped += 1
                print(f"[WARNING] Skipping quantized LoRA param: {name} (dtype: {param.dtype})")
                print(f"[WARNING] This LoRA parameter is quantized and cannot have gradients!")
        else:
            # Never touch quantized base weights - only set requires_grad on floating point
            if param.dtype.is_floating_point:
                param.requires_grad = False
    
    # If no LoRA parameters found, print all parameter names for debugging
    if trainable_count == 0:
        print("[ERROR] No LoRA parameters found! Printing all parameter names for debugging:")
        for i, name in enumerate(all_param_names[:20]):  # Print first 20 parameter names
            print(f"[DEBUG] Parameter {i+1}: {name}")
        if len(all_param_names) > 20:
            print(f"[DEBUG] ... and {len(all_param_names) - 20} more parameters")
        
        print("\n[DEBUG] Trying alternative LoRA detection patterns...")
        # Try broader patterns if standard ones fail
        broader_patterns = ['A', 'B', 'weight', 'bias']
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in broader_patterns):
                # Additional checks to avoid unfreezing all weights
                max_rank_threshold = config.get("max_lora_rank_threshold", 64)  # Configurable threshold
                if 'lora' in name.lower() or 'adapter' in name.lower() or (len(param.shape) == 2 and min(param.shape) <= max_rank_threshold):
                    # CRITICAL: Debug parameter details before setting gradients
                    print(f"[DEBUG] Checking parameter: {name}")
                    print(f"[DEBUG]   Shape: {param.shape}")
                    print(f"[DEBUG]   Dtype: {param.dtype}")
                    print(f"[DEBUG]   Is floating point: {param.dtype.is_floating_point}")
                    
                    # CRITICAL: Only set gradients on floating-point parameters
                    if param.dtype.is_floating_point:
                        param.requires_grad = True
                        trainable_count += 1
                        print(f"[DEBUG] ✅ Alternative LoRA param: {name} (shape: {param.shape}, dtype: {param.dtype})")
                    else:
                        quantized_params_skipped += 1
                        print(f"[WARNING] Skipping quantized alternative param: {name} (dtype: {param.dtype})")
                        print(f"[WARNING] This LoRA parameter is quantized and cannot have gradients!")
    
    if trainable_count == 0:
        print(f"\n[CRITICAL ERROR] No trainable LoRA parameters found!")
        print(f"[INFO] Total parameters analyzed: {len(all_param_names)}")
        print(f"[INFO] Quantized parameters skipped: {quantized_params_skipped}")
        
        if quantized_params_skipped > 0:
            print(f"[DIAGNOSIS] All LoRA parameters appear to be quantized.")
            print(f"[SOLUTION] Consider one of the following:")
            print(f"           1. Set load_in_4bit: false in config to disable quantization")
            print(f"           2. Use different LoRA injection that preserves floating-point parameters")
            print(f"           3. Check if ProbLoRA injection is compatible with quantization")
            print(f"[CONFIG] Current quantization setting: load_in_4bit = {config['load_in_4bit']}")
        
        raise RuntimeError("No trainable LoRA parameters found! Check ProbLoRA injection and parameter naming.")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Final parameter analysis and safety check
    print(f"\n[PARAMETER ANALYSIS] Final Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable parameter groups: {trainable_count}")
    print(f"  Quantized parameters skipped: {quantized_params_skipped}")
    print(f"  Trainable percentage: {100*trainable_params/total_params:.1f}%")
    
    print(f"[INFO] Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"[INFO] Found {trainable_count} trainable LoRA parameter groups")
    if quantized_params_skipped > 0:
        print(f"[INFO] Skipped {quantized_params_skipped} quantized parameters (cannot require gradients)")
    
    # Ensure model is in training mode
    model.train()
    print(f"[INFO] Model set to training mode: {model.training}")
    
    return model, tokenizer


def create_trainer(model, tokenizer, train_ds, val_ds, config, output_dir, tb_log_dir=None):
    """Create enhanced ARD-LoRA trainer with uncertainty evaluation and callbacks"""
    
    # Get ARD prior samples directly from config
    ard_prior_samples = config["ard_prior_samples"]
    print(f"[INFO] ARD Prior: Using {ard_prior_samples} samples for ARD prior estimation")
    
    # Ensure tokenizer consistency validation
    print(f"[TOKENIZER] Trainer Creation - Tokenizer Consistency Check:")
    print(f"[TOKENIZER]   Tokenizer name: {tokenizer.name_or_path}")
    print(f"[TOKENIZER]   PAD token ID: {tokenizer.pad_token_id}")
    print(f"[TOKENIZER]   EOS token ID: {tokenizer.eos_token_id}")
    print(f"[TOKENIZER]   PAD = EOS alignment: {tokenizer.pad_token_id == tokenizer.eos_token_id}")
    
    # Use the enhanced trainer builder that handles dataset splitting and callbacks
    trainer = build_clm_trainer(
        model=model,
        tokenizer=tokenizer,  # Ensure same tokenizer instance is passed
        train_dataset=train_ds,
        eval_dataset=val_ds,
        cfg=config,
        output_dir=output_dir,
        ard_prior_samples=ard_prior_samples,  # Pass absolute sample count directly
        enable_callbacks=config["enable_callbacks"],  # Enable ARD callbacks
        tb_log_dir=tb_log_dir  # Pass TensorBoard log directory
    )
    
    # Post-creation validation - ensure trainer uses the same tokenizer
    if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not tokenizer:
        print(f"[TOKENIZER] ⚠️  WARNING: Trainer tokenizer differs from input tokenizer!")
        print(f"[TOKENIZER]   Input tokenizer PAD ID: {tokenizer.pad_token_id}")
        print(f"[TOKENIZER]   Trainer tokenizer PAD ID: {trainer.tokenizer.pad_token_id}")
    else:
        print(f"[TOKENIZER] ✅ Trainer tokenizer consistency verified")
    
    # Validate data collator tokenizer consistency
    if hasattr(trainer, 'data_collator') and hasattr(trainer.data_collator, 'tokenizer'):
        if trainer.data_collator.tokenizer is not tokenizer:
            print(f"[TOKENIZER] ⚠️  WARNING: DataCollator tokenizer differs from input tokenizer!")
            print(f"[TOKENIZER]   Input tokenizer PAD ID: {tokenizer.pad_token_id}")
            print(f"[TOKENIZER]   DataCollator tokenizer PAD ID: {trainer.data_collator.tokenizer.pad_token_id}")
        else:
            print(f"[TOKENIZER] ✅ DataCollator tokenizer consistency verified")
    
    return trainer
def main():
    """Main training function with cached Bayesian-PEFT datasets"""
    print("=" * 80)
    print("ARD-LoRA Training with Cached Bayesian-PEFT Datasets")
    print("=" * 80)
    
    free_memory()
    
    # Load configuration from YAML file - no hardcoded defaults
    config = _merge_config({})  # Start with empty defaults, let YAML be the source of truth
    
    # Validate required configuration
    if not config:
        raise ValueError("No configuration found! Please ensure config.py imports a valid YAML configuration.")
    
    print(f"[CONFIG] Model: {config.get('model_name')}")
    print(f"[CONFIG] Dataset: {config.get('dataset_name')}")
    print(f"[CONFIG] Dataset Name: {config['dataset_name_specific']}")
    print(f"[CONFIG] KL Beta: {config.get('kl_loss_beta')}")
    print(f"[CONFIG] Rank: {config.get('rank')}")
    print(f"[CONFIG] Train Epochs: {config.get('train_epochs')}")
    
    # Memory optimization settings
    print(f"[CONFIG] Memory Optimizations for 40GB A100:")
    print(f"[CONFIG]   Use Cache: {config.get('use_cache')}")
    print(f"[CONFIG]   Gradient Checkpointing: {config.get('gradient_checkpointing')}")
    print(f"[CONFIG]   BF16: {config.get('bf16')}")
    print(f"[CONFIG]   Batch Size: {config.get('batch_size')}")
    print(f"[CONFIG]   Gradient Accumulation: {config.get('gradient_accumulation_steps')}")
    
    # ARD and uncertainty settings
    print(f"[CONFIG] ARD Prior Samples: {config.get('ard_prior_samples')}")
    print(f"[CONFIG] Uncertainty Eval Samples: {config.get('uncertainty_eval_samples')}")
    print(f"[CONFIG] Uncertainty ECE Bins: {config.get('uncertainty_n_bins')}")
    print(f"[CONFIG] Enable Callbacks: {config.get('enable_callbacks')}")
    print(f"[CONFIG] Enable Plotting: {config.get('enable_plotting')}")
    
    # Setup caching (Google Drive if available)
    cache_root = setup_cache_directory(config)
    config["cache_root"] = cache_root
    
    # Load model with ProbLoRA
    print("\n[STEP 1] Loading model and injecting ProbLoRA...")
    model, tokenizer = load_model_with_problora(config)
    
    # Load datasets with caching
    print(f"\n[STEP 2] Loading dataset with caching...")
    dataset_name = config["dataset_name_specific"]  # Which specific dataset
    
    try:
        train_ds, val_ds, tokenizer = load_bayesian_peft_with_caching(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer.name_or_path,
            config=config,
            cache_root=cache_root
        )
        
        # Validate tokenizer consistency after dataset loading
        print(f"[TOKENIZER] Post-dataset loading validation:")
        print(f"[TOKENIZER]   PAD token ID: {tokenizer.pad_token_id}")
        print(f"[TOKENIZER]   EOS token ID: {tokenizer.eos_token_id}")
        if tokenizer.pad_token_id != tokenizer.eos_token_id:
            print(f"[TOKENIZER] ⚠️  WARNING: PAD and EOS token IDs don't match after dataset loading!")
        else:
            print(f"[TOKENIZER] ✅ PAD and EOS alignment maintained: {tokenizer.pad_token_id}")
        
        print(f"[INFO] Training samples: {len(train_ds) if train_ds else 0}")
        print(f"[INFO] Validation samples: {len(val_ds) if val_ds else 0}")
        
        # Check validation dataset size for ARD prior estimation
        val_size = len(val_ds) if val_ds else 0
        ard_samples = config["ard_prior_samples"]
        if val_size > 0 and val_size < ard_samples:
            print(f"\n[WARNING] Validation dataset size issue:")
            print(f"          Validation samples: {val_size}")
            print(f"          ARD prior samples requested: {ard_samples}")
            print(f"          This may affect ARD prior estimation quality")
            print(f"          Consider increasing validation data or reducing ard_prior_samples in config")
            print(f"          ARD training will use all {val_size} validation samples for prior estimation")
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("[INFO] Please check that the Bayesian-PEFT repo is accessible")
        raise
    
    # Setup training
    print(f"\n[STEP 3] Setting up ARD-LoRA training...")
    
    # Create run-specific base directory
    base_output_dir = os.path.join(
        f"{config.get('model_name')}_ARD_LoRA_CLM_{dataset_name}"
    )
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get run-specific directories
    output_dir, model_ckpt_dir, tb_log_dir = get_output_dirs(
        config["runId"],
        base_output_dir
    )
    
    print(f"[INFO] Directory structure created:")
    print(f"       Base: {base_output_dir}")
    print(f"       Latent images: {output_dir}")
    print(f"       Model checkpoints: {model_ckpt_dir}")
    print(f"       TensorBoard logs: {tb_log_dir}")
    
    trainer = create_trainer(model, tokenizer, train_ds, val_ds, config, model_ckpt_dir, tb_log_dir)
    
    # Final tokenizer consistency validation before training
    print(f"\n[TOKENIZER] Final Pre-Training Validation:")
    print(f"[TOKENIZER]   Model tokenizer available: {hasattr(model, 'config') and hasattr(model.config, 'vocab_size')}")
    print(f"[TOKENIZER]   Main tokenizer PAD ID: {tokenizer.pad_token_id}")
    print(f"[TOKENIZER]   Trainer tokenizer PAD ID: {trainer.tokenizer.pad_token_id}")
    print(f"[TOKENIZER]   DataCollator tokenizer PAD ID: {trainer.data_collator.tokenizer.pad_token_id}")
    
    # Check if all tokenizers have the same PAD token ID
    all_pad_ids = [
        tokenizer.pad_token_id,
        trainer.tokenizer.pad_token_id,
        trainer.data_collator.tokenizer.pad_token_id
    ]
    if len(set(all_pad_ids)) == 1:
        print(f"[TOKENIZER] ✅ All components use consistent PAD token ID: {all_pad_ids[0]}")
    else:
        print(f"[TOKENIZER] ❌ ERROR: Inconsistent PAD token IDs found: {all_pad_ids}")
        raise ValueError("Tokenizer PAD token ID mismatch detected - this will cause training issues!")
    
    # Training with ARD
    print(f"\n[STEP 4] Starting training...")
    print(f"[INFO] Beta (KL strength): {config.get('kl_loss_beta')}")
    print(f"[INFO] Output directory: {model_ckpt_dir}")
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] TensorBoard logging enabled: {'tensorboard' in config['report_to']}")
    
    # Ensure model is in training mode before training starts
    model.train()
    print(f"[INFO] Model training mode verified: {model.training}")
    
    try:
        # Skip initial evaluation to avoid eval_dataset requirement
        print("\n[EVAL] Skipping initial evaluation - proceeding directly to training")
        
        last_checkpoint = None
        if os.path.isdir(model_ckpt_dir):
            checkpoints = [os.path.join(model_ckpt_dir, d) for d in os.listdir(model_ckpt_dir) if "checkpoint" in d]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getmtime)

        if last_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint}")
            print("\n[TRAIN] Starting ARD-LoRA training with uncertainty evaluation from checkpoint...")
            train_results = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print("Starting training from scratch...")
            print("\n[TRAIN] Starting ARD-LoRA training with uncertainty evaluation...")
            train_results = trainer.train()

        print(f"\n[SUCCESS] Training completed!")
        print(f"[INFO] Final loss: {train_results.training_loss:.4f}")
        
        # Print summary of uncertainty evolution
        if hasattr(trainer, 'uncertainty_results') and trainer.uncertainty_results:
            print(f"\n[UNCERTAINTY] Evolution across {len(trainer.uncertainty_results)} epochs:")
            for i, result in enumerate(trainer.uncertainty_results):
                epoch = result.get('epoch', i+1)
                acc = result.get('accuracy', 0)
                ece = result.get('ece', 0)
                nll = result.get('nll', 0)
                print(f"   Epoch {epoch}: ACC={acc:.4f}, ECE={ece:.4f}, NLL={nll:.4f}")
        
        # Save final model
        trainer.save_model()
        print(f"[SAVE] Model saved to {model_ckpt_dir}")
        
        # Print TensorBoard information
        if 'tensorboard' in config['report_to']:
            print(f"\n[TENSORBOARD] Training metrics logged to: {tb_log_dir}")
            print(f"[INFO] To view metrics, run: tensorboard --logdir {tb_log_dir}")
            print(f"[INFO] Metrics include:")
            print(f"       - train/ce_loss: Cross-entropy loss during training")
            print(f"       - train/kl_loss: KL divergence loss during training")
            print(f"       - train/total_loss: Total loss (CE + KL) during training")
            print(f"       - eval/ce_loss: Cross-entropy loss during evaluation")
            print(f"       - eval/kl_loss: KL divergence loss during evaluation")
            print(f"       - eval/total_loss: Total loss during evaluation")
            print(f"       - uncertainty/accuracy: Accuracy after each epoch")
            print(f"       - uncertainty/ece: Expected Calibration Error after each epoch")
            print(f"       - uncertainty/nll: Negative Log-Likelihood after each epoch")
        
        print("\n" + "=" * 80)
        print("✅ ARD-LoRA training with Bayesian-PEFT datasets completed successfully!")
        print(f"📁 Cached data available at: {cache_root}")
        print(f"📊 Model and logs saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
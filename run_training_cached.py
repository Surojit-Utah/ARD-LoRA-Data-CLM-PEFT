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
    model_name = merged.get("model_name", "LLaMA2-7B")
    if "models" in cfg and model_name in cfg["models"]:
        model_cfg = cfg["models"][model_name]
        merged.update(model_cfg.get("defaults", {}))
    
    # Apply dataset-specific config
    dataset_name = merged.get("dataset_name", "BayesianPEFT")
    if "datasets" in cfg and dataset_name in cfg["datasets"]:
        dataset_cfg = cfg["datasets"][dataset_name]
        merged.update(dataset_cfg)
    
    return merged


def setup_cache_directory(config):
    """
    Setup caching directory based on configuration.
    Uses cache_root from config or defaults to local cache.
    """
    cache_root = config.get("cache_root", "./data_cache")
    
    # Check if Google Drive path is available (for Colab usage)
    drive_cache = "/content/drive/MyDrive/ARD_LoRA_Data_Cache"
    if os.path.exists("/content/drive/MyDrive"):
        # Use Google Drive for persistent caching
        os.makedirs(drive_cache, exist_ok=True)
        print(f"[INFO] Using Google Drive cache: {drive_cache}")
        return drive_cache
    else:
        # Use local or configured cache directory
        os.makedirs(cache_root, exist_ok=True)
        print(f"[INFO] Using cache directory: {cache_root}")
        return cache_root


def load_model_with_problora(config):
    """Load LLaMA2 model and inject ProbLoRA layers"""
    model_name_or_path = config.get("model_name_or_path", "meta-llama/Llama-2-7b-hf")
    tokenizer_name = config.get("tokenizer_name", model_name_or_path)
    
    print(f"[INFO] Loading model: {model_name_or_path}")
    
    # Model loading arguments
    model_kwargs = {}
    if config.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    if config.get("fp16"):
        import torch
        model_kwargs["torch_dtype"] = torch.float16
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Inject ProbLoRA
    model = inject_problora_llama(
        model,
        rank=config.get("rank", 64),
        scaling=config.get("scaling", 1.0),
        prior_var=config.get("prior_var", 1.0),
        num_tokens=config.get("max_len", 2048),
        ard_prior_samples=config.get("ard_prior_samples", 1000)
    )
    
    # Freeze base parameters
    for name, param in model.named_parameters():
        if "A" in name or "B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return model, tokenizer


def create_trainer(model, tokenizer, train_ds, val_ds, config, output_dir, tb_log_dir=None):
    """Create enhanced ARD-LoRA trainer with uncertainty evaluation and callbacks"""
    
    # Update config to include TensorBoard log directory if provided
    if tb_log_dir:
        config["logging_dir"] = tb_log_dir
    
    # Use the enhanced trainer builder that handles dataset splitting and callbacks
    trainer = build_clm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        cfg=config,
        output_dir=output_dir,
        ard_prior_ratio=config.get("ard_prior_ratio", 0.5),  # Split ratio for ARD vs uncertainty eval
        enable_callbacks=config.get("enable_callbacks", True),  # Enable ARD callbacks
        tb_log_dir=tb_log_dir  # Pass TensorBoard log directory
    )
    
    return trainer
def main():
    """Main training function with cached Bayesian-PEFT datasets"""
    print("=" * 80)
    print("ARD-LoRA Training with Cached Bayesian-PEFT Datasets")
    print("=" * 80)
    
    free_memory()
    
    # Configuration
    defaults = {
        "model_name": "LLaMA2-7B",
        "dataset_name": "BayesianPEFT",
        "runId": 1,
        "rank": 16,
        "max_len": 2048,
        "kl_loss_beta": 0.01,
        "beta": 0.01,  # Alias for kl_loss_beta
        "prior_var": 1.0,
        "cache_root": "./data_cache",
        "ard_prior_ratio": 0.5,  # Split ratio: 0.5 for ARD, 0.5 for uncertainty eval
        "uncertainty_eval_samples": 1000,  # Number of samples for uncertainty evaluation
        "uncertainty_n_bins": 15,  # Number of bins for ECE calculation
        "train_epochs": 3,  # Multiple epochs to see uncertainty evolution
        "batch_size": 4,
        "gradient_accumulation_steps": 32,
        "learning_rate": 1e-5,
        "report_to": ["tensorboard"],  # Log uncertainty metrics to tensorboard
        # Callback configuration
        "enable_callbacks": True,  # Enable ARD callbacks
        "ard_prior_samples": 1000,  # Samples for ARD prior estimation
        "enable_plotting": True,  # Enable latent plotting
        "enable_resampling": False,  # Enable held-out resampling (disabled for CLM)
        "plot_start_epoch": 2,  # Start plotting from epoch 2
        "plot_interval": 2  # Plot every 2 epochs
    }
    
    config = _merge_config(defaults)
    
    print(f"[CONFIG] Model: {config.get('model_name')}")
    print(f"[CONFIG] Dataset: {config.get('dataset_name')}")
    print(f"[CONFIG] Dataset Name: {config.get('dataset_name_specific', 'alpaca')}")
    print(f"[CONFIG] KL Beta: {config.get('kl_loss_beta')}")
    print(f"[CONFIG] Rank: {config.get('rank')}")
    print(f"[CONFIG] Train Epochs: {config.get('train_epochs')}")
    print(f"[CONFIG] ARD Prior Ratio: {config.get('ard_prior_ratio')}")
    print(f"[CONFIG] Uncertainty Eval Samples: {config.get('uncertainty_eval_samples')}")
    print(f"[CONFIG] Uncertainty ECE Bins: {config.get('uncertainty_n_bins')}")
    print(f"[CONFIG] Enable Callbacks: {config.get('enable_callbacks')}")
    print(f"[CONFIG] ARD Prior Samples: {config.get('ard_prior_samples')}")
    print(f"[CONFIG] Enable Plotting: {config.get('enable_plotting')}")
    
    # Setup caching (Google Drive if available)
    cache_root = setup_cache_directory(config)
    config["cache_root"] = cache_root
    
    # Load model with ProbLoRA
    print("\n[STEP 1] Loading model and injecting ProbLoRA...")
    model, tokenizer = load_model_with_problora(config)
    
    # Load datasets with caching
    print(f"\n[STEP 2] Loading dataset with caching...")
    dataset_name = config.get("dataset_name_specific", "alpaca")  # Which specific dataset
    
    try:
        train_ds, val_ds, tokenizer = load_bayesian_peft_with_caching(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer.name_or_path,
            config=config,
            cache_root=cache_root
        )
        
        print(f"[INFO] Training samples: {len(train_ds) if train_ds else 0}")
        print(f"[INFO] Validation samples: {len(val_ds) if val_ds else 0}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("[INFO] Please check that the Bayesian-PEFT repo is accessible")
        raise
    
    # Setup training
    print(f"\n[STEP 3] Setting up ARD-LoRA training...")
    
    # Create run-specific base directory
    base_output_dir = os.path.join(
        os.getcwd(), 
        "run_outputs",
        f"ARD_LoRA_{config.get('model_name')}_{dataset_name}"
    )
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get run-specific directories
    output_dir, model_ckpt_dir, tb_log_dir = get_output_dirs(
        config.get("runId", 1),
        base_output_dir
    )
    
    print(f"[INFO] Directory structure created:")
    print(f"       Base: {base_output_dir}")
    print(f"       Latent images: {output_dir}")
    print(f"       Model checkpoints: {model_ckpt_dir}")
    print(f"       TensorBoard logs: {tb_log_dir}")
    
    trainer = create_trainer(model, tokenizer, train_ds, val_ds, config, model_ckpt_dir, tb_log_dir)
    
    # Training with ARD
    print(f"\n[STEP 4] Starting training...")
    print(f"[INFO] Beta (KL strength): {config.get('kl_loss_beta')}")
    print(f"[INFO] Output directory: {model_ckpt_dir}")
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] TensorBoard logging enabled: {'tensorboard' in config.get('report_to', [])}")
    
    try:
        # Initial evaluation
        if val_ds:
            print("\n[EVAL] Initial validation...")
            eval_results = trainer.evaluate()
            print(f"[EVAL] Initial loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        
        # Train with automatic uncertainty evaluation after each epoch
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
        if 'tensorboard' in config.get('report_to', []):
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
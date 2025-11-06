"""
Bayesian-PEFT Dataset Integration with Google Drive Caching for CLM
===================================================================

This module provides ARD-LoRA compatible dataset loading with Google Drive caching
for persistent access across training runs.

Strategy:
1. Use HuggingFace datasets directly (following Bayesian-PEFT approach)
2. Download and process datasets using standard loaders
3. Cache processed datasets in Google Drive 
4. Provide ARD-LoRA compatible interface with 3-way data flow
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer


class BayesianPEFTDataManager:
    """
    Manager for Bayesian-PEFT datasets with Google Drive caching.
    Downloads data once and caches in Google Drive for future use.
    """
    
    def __init__(self, cache_root: str = "/content/drive/MyDrive/ARD_LoRA_Data_Cache"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_cache_path(self, dataset_name: str, split: str = "train") -> Path:
        """Get cache path for a specific dataset split"""
        return self.cache_root / f"{dataset_name}_{split}.json"
    
    def is_cached(self, dataset_name: str) -> bool:
        """Check if dataset is already cached"""
        train_cache = self.get_dataset_cache_path(dataset_name, "train")
        val_cache = self.get_dataset_cache_path(dataset_name, "validation")
        
        # Consider cached if both train and validation files exist
        both_exist = train_cache.exists() and val_cache.exists()
        
        if both_exist:
            print(f"[CACHE] Found cached {dataset_name}: train={train_cache.exists()}, val={val_cache.exists()}")
        
        return both_exist
    
    def cache_dataset(self, dataset_name: str, data: Dict[str, Any]):
        """Cache dataset to Google Drive storage"""
        for split, dataset in data.items():
            cache_path = self.get_dataset_cache_path(dataset_name, split)
            
            if hasattr(dataset, 'to_dict'):
                # HuggingFace Dataset
                dataset_dict = dataset.to_dict()
            elif isinstance(dataset, list):
                # List of examples
                dataset_dict = {"examples": dataset}
            else:
                # Custom dataset - extract examples
                dataset_dict = {"examples": list(dataset)}
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_dict, f, ensure_ascii=False, indent=2)
            
            print(f"[CACHE] Saved {len(dataset_dict.get('examples', dataset_dict.get('input_ids', [])))} examples to {cache_path}")
    
    def load_cached_dataset(self, dataset_name: str) -> Dict[str, HFDataset]:
        """Load dataset from cache"""
        cached_data = {}
        
        for split in ["train", "validation", "test"]:
            cache_path = self.get_dataset_cache_path(dataset_name, split)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "examples" in data:
                    # Convert back to HuggingFace Dataset
                    cached_data[split] = HFDataset.from_list(data["examples"])
                else:
                    # Direct HF Dataset format
                    cached_data[split] = HFDataset.from_dict(data)
                
                print(f"[CACHE] Loaded {len(cached_data[split])} examples from {cache_path}")
        
        # Check if we have both train and validation cached - if so, use them directly
        if "train" in cached_data and "validation" in cached_data:
            print(f"[CACHE] ✅ Using existing cached train/validation split (train={len(cached_data['train'])}, val={len(cached_data['validation'])})")
            return cached_data
        
        # Only create validation split if we ONLY have training data cached
        elif "train" in cached_data and "validation" not in cached_data and len(cached_data["train"]) > 0:
            print(f"[INFO] Found cached training data without validation. Creating validation split...")
            
            train_ds = cached_data["train"]
            split_data = train_ds.train_test_split(test_size=0.1, seed=42)
            cached_data["train"] = split_data['train']
            cached_data["validation"] = split_data['test']
            
            print(f"[INFO] Created validation split from cached training data - Train: {len(cached_data['train'])}, Validation: {len(cached_data['validation'])}")
            
            # Cache the new validation split for future use
            self.cache_dataset(dataset_name, {"validation": cached_data["validation"]})
            # Update the train cache with the reduced training set
            self.cache_dataset(dataset_name, {"train": cached_data["train"]})
        
        return cached_data
    
    def download_and_cache_dataset(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """
        Download dataset using Bayesian-PEFT approach and cache locally.
        This mimics how they construct datasets in their repository.
        """
        # Double-check cache before downloading (defensive programming)
        if self.is_cached(dataset_name):
            print(f"[CACHE] Dataset {dataset_name} found in cache during download attempt, using cached version...")
            return self.load_cached_dataset(dataset_name)
        
        print(f"[DOWNLOAD] Fetching {dataset_name} using Bayesian-PEFT approach...")
        
        try:
            # Import HuggingFace datasets directly (mimicking their approach)
            from datasets import load_dataset
            
            # Load dataset based on dataset name (following their S2ClassDataset.py approach)
            if dataset_name.lower() == "sst2":
                # Load SST-2 from GLUE benchmark (like they do in S2ClassDataset.py line 42)
                raw_dataset = load_dataset("glue", "sst2")
                print(f"[INFO] Loaded SST-2 from GLUE: {len(raw_dataset['train'])} train, {len(raw_dataset['validation'])} validation")
                
                # Process the dataset following their approach
                def process_sst2_sample(example):
                    return {
                        "sentence": example["sentence"],
                        "label": example["label"],
                        "text": example["sentence"],  # For compatibility
                        "full_text": example["sentence"],  # For our processing
                        "prompt_text": example["sentence"]  # For prompt construction
                    }
                
                # ALWAYS use only training data and create our own validation split
                # Combine all available data and split from training data only
                all_train_data = raw_dataset["train"].map(process_sst2_sample)
                print(f"[INFO] Creating validation split from training data only (ignoring natural validation split)")
                split_data = all_train_data.train_test_split(test_size=0.1, seed=42)
                train_data = split_data['train']
                val_data = split_data['test']
                
                datasets = {"train": train_data, "validation": val_data}
                
            elif dataset_name.lower() in ["piqa", "hellaswag", "winogrande_s", "winogrande_m", "arc_easy", "arc_challenge", "boolq", "anli", "rte", "cb", "copa", "glue_mrpc"]:
                # Handle other classification datasets following their approach
                if dataset_name.lower() == "piqa":
                    raw_dataset = load_dataset("piqa")
                elif dataset_name.lower() == "hellaswag":
                    raw_dataset = load_dataset("hellaswag")
                elif dataset_name.lower() in ["winogrande_s", "winogrande_m"]:
                    # Support winogrande subset parameter from config
                    subset_map = {
                        "small": "winogrande_s",
                        "medium": "winogrande_m", 
                        "large": "winogrande_l",
                        "xl": "winogrande_xl"
                    }
                    # Determine subset from dataset name
                    if dataset_name.lower() == "winogrande_s":
                        subset = "small"
                    elif dataset_name.lower() == "winogrande_m":
                        subset = "medium"
                    subset_name = subset_map.get(subset)
                    print(f"[INFO] Loading WinoGrande subset: {subset} -> {subset_name}")
                    raw_dataset = load_dataset("winogrande", subset_name)
                elif dataset_name.lower() == "arc_easy":
                    raw_dataset = load_dataset("ai2_arc", "ARC-Easy")
                elif dataset_name.lower() == "arc_challenge":
                    raw_dataset = load_dataset("ai2_arc", "ARC-Challenge")
                elif dataset_name.lower() == "boolq":
                    raw_dataset = load_dataset("super_glue", "boolq")
                elif dataset_name.lower() == "rte":
                    raw_dataset = load_dataset("glue", "rte")
                elif dataset_name.lower() == "glue_mrpc":
                    raw_dataset = load_dataset("glue", "mrpc")
                elif dataset_name.lower() == "cb":
                    raw_dataset = load_dataset("super_glue", "cb")
                elif dataset_name.lower() == "copa":
                    raw_dataset = load_dataset("super_glue", "copa")
                else:
                    raw_dataset = load_dataset(dataset_name)
                
                print(f"[INFO] Loaded {dataset_name}: {raw_dataset}")
                
                # Create basic processing function
                def process_classification_sample(example):
                    # Extract main text field (varies by dataset)
                    if "sentence" in example:
                        text = example["sentence"]
                    elif "sentence1" in example and "sentence2" in example:
                        # Handle MRPC-style dual sentence format
                        text = f"{example['sentence1']} {example['sentence2']}"
                    elif "question" in example:
                        text = example["question"]
                    elif "premise" in example:
                        text = f"{example['premise']} {example.get('hypothesis', '')}"
                    else:
                        text = str(example)
                    
                    return {
                        "text": text,
                        "label": example.get("label", 0),
                        "full_text": text,
                        "prompt_text": text
                    }
                
                # Get train and validation splits
                train_split = "train" if "train" in raw_dataset else list(raw_dataset.keys())[0]
                
                # ALWAYS create validation from training data only (ignore natural validation splits)
                train_data = raw_dataset[train_split].map(process_classification_sample)
                print(f"[INFO] Creating validation split from training data only (ignoring any natural validation split)")
                split_data = train_data.train_test_split(test_size=0.1, seed=42)
                train_data = split_data['train']
                val_data = split_data['test']
                
                datasets = {"train": train_data, "validation": val_data}
                
            else:
                # Generic dataset loading
                raw_dataset = load_dataset(dataset_name)
                print(f"[INFO] Loaded {dataset_name}: {raw_dataset}")
                
                # Simple processing
                def process_generic_sample(example):
                    text = str(list(example.values())[0])  # Use first field as text
                    return {
                        "text": text,
                        "label": example.get("label", 0),
                        "full_text": text,
                        "prompt_text": text
                    }
                
                # Process splits - ALWAYS create validation from training data only
                if "train" in raw_dataset:
                    train_data = raw_dataset["train"].map(process_generic_sample)
                    print(f"[INFO] Creating validation split from training data only (ignoring any natural validation split)")
                    split_data = train_data.train_test_split(test_size=0.1, seed=42)
                    train_data = split_data['train']
                    val_data = split_data['test']
                    
                    datasets = {"train": train_data, "validation": val_data}
                else:
                    # Single split dataset
                    all_data = raw_dataset[list(raw_dataset.keys())[0]].map(process_generic_sample)
                    split_data = all_data.train_test_split(test_size=0.1, seed=42)
                    datasets = {"train": split_data['train'], "validation": split_data['test']}
            
            # Cache the processed datasets
            self.cache_dataset(dataset_name, datasets)
            
            print(f"[SUCCESS] Processed {dataset_name} - Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")
            return self.load_cached_dataset(dataset_name)
            
        except Exception as e:
            print(f"[ERROR] Failed to download {dataset_name}: {e}")
            raise ValueError(f"Could not load dataset {dataset_name}. Please check dataset name and availability.")
    
    def get_dataset(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """
        Main method: Get dataset with caching.
        Downloads once, then uses cache for subsequent calls.
        """
        if self.is_cached(dataset_name):
            print(f"[CACHE] ✅ Loading {dataset_name} from cache (no download needed)...")
            return self.load_cached_dataset(dataset_name)
        else:
            print(f"[DOWNLOAD] ⬇️ {dataset_name} not cached, downloading and caching...")
            return self.download_and_cache_dataset(dataset_name, config)


class ARDLoRADatasetWrapper:
    """
    Wrapper that makes Bayesian-PEFT datasets compatible with ARD-LoRA training.
    Handles tokenization and formatting for causal language modeling.
    """
    
    def __init__(self, dataset_name: str, tokenizer_name: str, config: Dict[str, Any], cache_root: str = "/content/drive/MyDrive/ARD_LoRA_Data_Cache"):
        self.dataset_name = dataset_name
        self.config = config
        self.cache_root = cache_root
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize dataset manager
        self.data_manager = BayesianPEFTDataManager(cache_root=cache_root)
        
        # Load/download dataset
        self.datasets = self.data_manager.get_dataset(dataset_name, config)
    
    def get_processed_datasets(self) -> Tuple[Optional[HFDataset], Optional[HFDataset]]:
        """
        Get tokenized datasets ready for ARD-LoRA training.
        Returns (train_dataset, validation_dataset)
        
        Data flow:
        1. Training data -> FULL training dataset (used for both SGD and dynamic ARD sampling each epoch)
        2. Validation data -> kept fixed for evaluation only
        3. ARD samples -> dynamically resampled from train_ds each epoch (not pre-split)
        """
        train_ds = self.datasets.get("train")
        val_ds = self.datasets.get("validation")
        
        # Process FULL training dataset (no splitting - used for both SGD and ARD)
        if train_ds is not None:
            train_ds = self._process_dataset(train_ds)
            print(f"[INFO] Training: {len(train_ds)} samples (for both SGD and dynamic ARD sampling)")
        
        # Process validation dataset (fixed, for evaluation only)
        if val_ds is not None:
            val_ds = self._process_dataset(val_ds)
            print(f"[INFO] Validation: {len(val_ds)} samples (for evaluation only)")
        
        return train_ds, val_ds
    
    def _process_dataset(self, dataset: HFDataset) -> HFDataset:
        """Process dataset for causal LM with prompt masking"""
        max_len = self.config.get("max_len", 2048)
        
        def tokenize_and_mask(batch):
            # Handle different input formats
            if "full_text" in batch:
                full_texts = batch["full_text"]
                prompt_texts = batch.get("prompt_text", [""] * len(full_texts))
            elif "instruction" in batch and "output" in batch:
                # Create full texts from instruction/output
                full_texts = []
                prompt_texts = []
                for i in range(len(batch["instruction"])):
                    instruction = batch["instruction"][i]
                    input_text = batch.get("input", [""] * len(batch["instruction"]))[i]
                    output = batch["output"][i]
                    
                    prompt = instruction
                    if input_text:
                        prompt += f"\n{input_text}"
                    full_text = prompt + f"\n{output}"
                    
                    prompt_texts.append(prompt)
                    full_texts.append(full_text)
            else:
                raise ValueError("Unsupported dataset format")
            
            # Tokenize full sequences
            full_tok = self.tokenizer(
                full_texts,
                truncation=True,
                max_length=max_len,
                padding="max_length"
            )
            
            # Tokenize prompts (for masking)
            prompt_tok = self.tokenizer(
                prompt_texts,
                truncation=True,
                max_length=max_len
            )
            
            # Create labels with prompt masking
            labels = []
            for i, (full_ids, prompt_ids) in enumerate(zip(full_tok["input_ids"], prompt_tok["input_ids"])):
                label = full_ids.copy()
                # Mask prompt tokens
                prompt_len = len(prompt_ids)
                for j in range(min(prompt_len, len(label))):
                    label[j] = -100
                labels.append(label)
            
            full_tok["labels"] = labels
            return full_tok
        
        return dataset.map(
            tokenize_and_mask,
            batched=True,
            remove_columns=dataset.column_names
        )


def load_bayesian_peft_with_caching(dataset_name: str, tokenizer_name: str, config: Dict[str, Any], cache_root: str = "/content/drive/MyDrive/ARD_LoRA_Data_Cache"):
    """
    Main function to load Bayesian-PEFT datasets with Google Drive caching.
    
    Args:
        dataset_name: Name of dataset (e.g., "alpaca", "dolly", "gsm8k")
        tokenizer_name: HuggingFace tokenizer name
        config: Dataset configuration
        cache_root: Google Drive cache directory
    
    Returns:
        (train_dataset, val_dataset, tokenizer)
        - train_dataset: Full training data for both SGD and dynamic ARD sampling
        - val_dataset: Fixed validation data for evaluation only
        - tokenizer: Configured tokenizer
    """
    wrapper = ARDLoRADatasetWrapper(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        config=config,
        cache_root=cache_root
    )
    
    train_ds, val_ds = wrapper.get_processed_datasets()
    
    # Apply validation dataset size cap
    max_val_samples = config.get("max_validation_samples", 5000)
    if val_ds is not None and len(val_ds) > max_val_samples:
        print(f"[INFO] Capping validation dataset from {len(val_ds)} to {max_val_samples} samples")
        # Randomly sample to maintain data diversity
        import random
        indices = list(range(len(val_ds)))
        random.seed(42)  # For reproducibility
        selected_indices = random.sample(indices, max_val_samples)
        val_ds = val_ds.select(selected_indices)
        print(f"[INFO] Validation dataset capped to {len(val_ds)} samples")
    
    # Print final dataset summary
    print(f"\n[DATASET SUMMARY]")
    print(f"  Training (SGD + ARD): {len(train_ds) if train_ds else 0} samples")
    print(f"  Validation (eval): {len(val_ds) if val_ds else 0} samples") 
    print(f"  Total samples: {(len(train_ds) if train_ds else 0) + (len(val_ds) if val_ds else 0)}")
    print(f"[INFO] ARD samples will be dynamically sampled from training data each epoch")
    print(f"[INFO] Validation samples are fixed and used ONLY for evaluation")
    
    return train_ds, val_ds, wrapper.tokenizer

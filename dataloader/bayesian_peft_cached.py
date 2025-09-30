"""
Bayesian-PEFT Dataset Integration with Local Caching for CLM
===========================================================

This module leverages Bayesian-PEFT's dataset classes while ensuring
data is downloaded and cached locally (e.g., Google Drive) for 
persistent access across training runs.

Strategy:
1. Clone Bayesian-PEFT repo for dataset utilities
2. Download datasets using their loaders
3. Cache processed datasets locally 
4. Provide ARD-LoRA compatible interface
"""

import os
import sys
import json
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer


class BayesianPEFTDataManager:
    """
    Manager for Bayesian-PEFT datasets with local caching.
    Downloads data once and caches for future use.
    """
    
    def __init__(self, cache_root: str = "./data_cache", repo_path: str = "./external/bayesian-peft"):
        self.cache_root = Path(cache_root)
        self.repo_path = Path(repo_path)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        # Ensure Bayesian-PEFT repo is available
        self._setup_repo()
    
    def _setup_repo(self):
        """Clone Bayesian-PEFT repo if not exists"""
        if not self.repo_path.exists():
            print(f"[INFO] Cloning Bayesian-PEFT repo to {self.repo_path}")
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "git", "clone", 
                "https://github.com/Wang-ML-Lab/bayesian-peft",
                str(self.repo_path)
            ], check=True)
        
        # Add to Python path for imports
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
    
    def get_dataset_cache_path(self, dataset_name: str, split: str = "train") -> Path:
        """Get cache path for a specific dataset split"""
        return self.cache_root / f"{dataset_name}_{split}.json"
    
    def is_cached(self, dataset_name: str) -> bool:
        """Check if dataset is already cached"""
        train_cache = self.get_dataset_cache_path(dataset_name, "train")
        return train_cache.exists()
    
    def cache_dataset(self, dataset_name: str, data: Dict[str, Any]):
        """Cache dataset to local storage"""
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
        
        # If we have train data but no validation data, create validation split from train
        if "train" in cached_data and "validation" not in cached_data and len(cached_data["train"]) > 0:
            print(f"[INFO] No cached validation data found. Creating validation split from cached train data...")
            
            train_ds = cached_data["train"]
            split_data = train_ds.train_test_split(test_size=0.1, seed=42)
            cached_data["train"] = split_data['train']
            cached_data["validation"] = split_data['test']
            
            print(f"[INFO] Created validation split from cache - Train: {len(cached_data['train'])}, Validation: {len(cached_data['validation'])}")
            
            # Cache the new validation split for future use
            self.cache_dataset(dataset_name, {"validation": cached_data["validation"]})
            # Update the train cache with the reduced training set
            self.cache_dataset(dataset_name, {"train": cached_data["train"]})
        
        return cached_data
    
    def download_and_cache_dataset(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """
        Download dataset using Bayesian-PEFT loaders and cache locally.
        This is where we leverage their dataset classes.
        """
        print(f"[DOWNLOAD] Fetching {dataset_name} using Bayesian-PEFT loaders...")
        
        try:
            # Import their dataset utilities
            from dataset.utils import get_dataset  # From their repo
            from dataset.S2SDataset import S2SDataset
            from dataset.S2ClassDataset import S2ClassDataset
            
            # Use their dataset factory based on config
            dataset_class = config.get("dataset_class", "S2SDataset")
            
            if dataset_class == "S2SDataset":
                dataset_loader = S2SDataset(
                    dataset_name=dataset_name,
                    max_length=config.get("max_len", 2048),
                    **config
                )
            elif dataset_class == "S2ClassDataset":
                dataset_loader = S2ClassDataset(
                    dataset_name=dataset_name,
                    max_length=config.get("max_len", 512),
                    **config
                )
            else:
                raise ValueError(f"Unsupported dataset class: {dataset_class}")
            
            # Get train/val splits using their methods
            train_data = dataset_loader.get_train_data()
            val_data = dataset_loader.get_validation_data() if hasattr(dataset_loader, 'get_validation_data') else None
            
            # If no validation data, create from train split
            if val_data is None and train_data is not None:
                print(f"[INFO] No validation split found for {dataset_name}. Creating from train data...")
                # Split train data: 90% train, 10% validation
                train_size = int(0.9 * len(train_data))
                val_size = len(train_data) - train_size
                
                # Create train/val split
                from datasets import Dataset
                if isinstance(train_data, Dataset):
                    # For HuggingFace datasets
                    split_data = train_data.train_test_split(test_size=0.1, seed=42)
                    train_data = split_data['train']
                    val_data = split_data['test']
                elif hasattr(train_data, '__len__') and hasattr(train_data, '__getitem__'):
                    # For list-like data
                    import random
                    indices = list(range(len(train_data)))
                    random.seed(42)
                    random.shuffle(indices)
                    
                    train_indices = indices[:train_size]
                    val_indices = indices[train_size:]
                    
                    if hasattr(train_data, 'select'):
                        train_data = train_data.select(train_indices)
                        val_data = train_data.select(val_indices)
                    else:
                        # Create new lists
                        original_data = train_data
                        train_data = [original_data[i] for i in train_indices]
                        val_data = [original_data[i] for i in val_indices]
                        
                        # Convert to HF datasets if possible
                        from datasets import Dataset
                        train_data = Dataset.from_list(train_data)
                        val_data = Dataset.from_list(val_data)
                
                print(f"[INFO] Created validation split: {len(train_data)} train, {len(val_data)} validation")
            
            # Convert to our format
            datasets = {"train": train_data}
            if val_data is not None:
                datasets["validation"] = val_data
            
            # Cache for future use
            self.cache_dataset(dataset_name, datasets)
            
            return self.load_cached_dataset(dataset_name)
            
        except Exception as e:
            print(f"[ERROR] Failed to download {dataset_name}: {e}")
            print("[FALLBACK] Using manual download approach...")
            return self._fallback_download(dataset_name, config)
    
    def _fallback_download(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """Fallback download method if their loaders fail"""
        # This would implement direct downloads for known datasets
        if dataset_name.lower() == "alpaca":
            return self._download_alpaca(config)
        elif dataset_name.lower() == "dolly":
            return self._download_dolly(config)
        else:
            raise ValueError(f"No fallback available for dataset: {dataset_name}")
    
    def _download_alpaca(self, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """Download Alpaca dataset directly"""
        from datasets import load_dataset
        
        print("[FALLBACK] Downloading Alpaca dataset from HuggingFace...")
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # Convert to our format
        def format_alpaca(example):
            instruction = example["instruction"]
            input_text = example["input"] if example["input"] else ""
            output_text = example["output"]
            
            prompt = instruction
            if input_text:
                prompt += f"\n{input_text}"
            full_text = prompt + f"\n{output_text}"
            
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "prompt_text": prompt,
                "full_text": full_text
            }
        
        train_ds = dataset["train"].map(format_alpaca)
        
        # Create validation split from train data
        print("[INFO] Creating train/validation split for Alpaca...")
        split_data = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split_data['train']
        val_ds = split_data['test']
        
        print(f"[INFO] Alpaca splits - Train: {len(train_ds)}, Validation: {len(val_ds)}")
        
        datasets = {"train": train_ds, "validation": val_ds}
        self.cache_dataset("alpaca", datasets)
        
        return datasets
    
    def _download_dolly(self, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """Download Dolly dataset directly"""
        from datasets import load_dataset
        
        print("[FALLBACK] Downloading Dolly dataset from HuggingFace...")
        dataset = load_dataset("databricks/databricks-dolly-15k")
        
        def format_dolly(example):
            instruction = example["instruction"]
            context = example["context"] if example["context"] else ""
            response = example["response"]
            
            prompt = instruction
            if context:
                prompt += f"\nContext: {context}"
            full_text = prompt + f"\n{response}"
            
            return {
                "instruction": instruction,
                "context": context,
                "response": response,
                "prompt_text": prompt,
                "full_text": full_text
            }
        
        train_ds = dataset["train"].map(format_dolly)
        
        # Create validation split from train data
        print("[INFO] Creating train/validation split for Dolly...")
        split_data = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split_data['train']
        val_ds = split_data['test']
        
        print(f"[INFO] Dolly splits - Train: {len(train_ds)}, Validation: {len(val_ds)}")
        
        datasets = {"train": train_ds, "validation": val_ds}
        self.cache_dataset("dolly", datasets)
        
        return datasets
    
    def get_dataset(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, HFDataset]:
        """
        Main method: Get dataset with caching.
        Downloads once, then uses cache for subsequent calls.
        """
        if self.is_cached(dataset_name):
            print(f"[INFO] Loading {dataset_name} from cache...")
            return self.load_cached_dataset(dataset_name)
        else:
            print(f"[INFO] Downloading and caching {dataset_name}...")
            return self.download_and_cache_dataset(dataset_name, config)


class ARDLoRADatasetWrapper:
    """
    Wrapper that makes Bayesian-PEFT datasets compatible with ARD-LoRA training.
    Handles tokenization and formatting for causal language modeling.
    """
    
    def __init__(self, dataset_name: str, tokenizer_name: str, config: Dict[str, Any], cache_root: str = "./data_cache"):
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
        """
        train_ds = self.datasets.get("train")
        val_ds = self.datasets.get("validation")
        
        if train_ds is not None:
            train_ds = self._process_dataset(train_ds)
        
        if val_ds is not None:
            val_ds = self._process_dataset(val_ds)
        
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


def load_bayesian_peft_with_caching(dataset_name: str, tokenizer_name: str, config: Dict[str, Any], cache_root: str = "./data_cache"):
    """
    Main function to load Bayesian-PEFT datasets with local caching.
    
    Args:
        dataset_name: Name of dataset (e.g., "alpaca", "dolly", "gsm8k")
        tokenizer_name: HuggingFace tokenizer name
        config: Dataset configuration
        cache_root: Local cache directory (can be Google Drive path)
    
    Returns:
        (train_dataset, val_dataset, tokenizer)
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
    
    return train_ds, val_ds, wrapper.tokenizer
"""
ARC-Easy Dataset Loader for ARD-LoRA
=====================================

This module provides ARC-Easy dataset loading compatible with ARD-LoRA's
classification-based training approach (single-token prediction).

Follows the BayesianPEFT approach:
1. Format prompts with question and multiple-choice answers
2. Tokenize with proper padding (left-padding for causal LM)
3. Extract target token IDs for valid answers (A, B, C, D, E)
4. Return batches with (inputs, classes, targets)
"""

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple
from collections import OrderedDict


class ARCEasyDataset:
    """
    ARC-Easy dataset loader for classification-style training on causal LM.
    
    Key Features:
    - Formats questions with multiple-choice options
    - Uses last-token prediction (not full sequence generation)
    - Filters logits to only valid answer tokens (A, B, C, D, E)
    - Compatible with ARD-LoRA's training loop
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any] = None,
        add_space: bool = None,
        max_seq_len: int = None,
        few_shot: bool = None
    ):
        """
        Initialize ARC-Easy dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer for the model
            config: Configuration dictionary (from YAML)
            add_space: Whether to add leading space to answer tokens (overrides config)
            max_seq_len: Maximum sequence length for tokenization (overrides config)
            few_shot: Whether to use few-shot or zero-shot prompting (overrides config)
        """
        # Load config values with fallbacks
        if config is not None:
            self.add_space = add_space if add_space is not None else config.get('add_space_to_answers')
            self.max_seq_len = max_seq_len if max_seq_len is not None else config.get('max_len')
            self.few_shot = few_shot if few_shot is not None else config.get('few_shot_prompting')
        else:
            self.add_space = add_space if add_space is not None else True
            self.max_seq_len = max_seq_len if max_seq_len is not None else 512
            self.few_shot = few_shot if few_shot is not None else False
        
        print(f"[ARC-EASY] Loading dataset from HuggingFace...")
        self.dset = load_dataset("allenai/ai2_arc", "ARC-Easy")
        
        self.tokenizer = tokenizer
        self.n_labels = 5  # A, B, C, D, E (max 5 choices)
        
        # Set padding side to left for causal LM (important!)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Choose prompt template
        self.preamble = self.few_shot_preamble if self.few_shot else self.zero_shot_preamble
        
        # Compute target token IDs for answer labels
        self._compute_target_ids()
        
        print(f"[ARC-EASY] Dataset loaded:")
        print(f"[ARC-EASY]   Train: {len(self.dset['train'])} examples")
        print(f"[ARC-EASY]   Validation: {len(self.dset['validation'])} examples")
        print(f"[ARC-EASY]   Test: {len(self.dset['test'])} examples")
        print(f"[ARC-EASY]   Target IDs: {self.target_ids.tolist()}")
    
    # Prompt templates (from BayesianPEFT)
    few_shot_preamble = """Return the label of the correct answer for each question below.

Which two body systems are directly involved in movement?
Choices:
A) muscular and skeletal
B) digestive and muscular
C) skeletal and respiratory
E) respiratory and digestive
Answer: A

{question}
Choices:
{choices}
Answer:"""
    
    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""
    
    def _compute_target_ids(self):
        """
        Compute token IDs for valid answer labels (A, B, C, D, E).
        
        CRITICAL: This assumes each answer tokenizes to a single token!
        The add_space parameter controls whether we use " A" or "A".
        """
        spc = " " if self.add_space else ""
        labels = [f"{spc}{chr(ord('A') + i)}" for i in range(self.n_labels)]
        # labels = [" A", " B", " C", " D", " E"] if add_space=True
        
        # Tokenize labels
        tokenized = self.tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        # Extract last token (assumes single-token encoding!)
        self.target_ids = tokenized.input_ids[:, -1:]
        
        # Create label<->target mappings
        self.label2target = OrderedDict(
            [(i, self.target_ids[i]) for i in range(self.n_labels)]
        )
        self.target2label = OrderedDict(
            [(self.target_ids[i].item(), i) for i in range(self.n_labels)]
        )
        
        print(f"[ARC-EASY] Target token IDs computed:")
        for i, label in enumerate(labels):
            print(f"[ARC-EASY]   {repr(label)} -> token_id={self.target_ids[i].item()}")
    
    def _format_prompts(self, batch: List[Dict]) -> List[str]:
        """
        Format prompts for a batch of examples.
        
        Args:
            batch: List of examples from the dataset
        
        Returns:
            List of formatted prompt strings
        """
        prompts = []
        for example in batch:
            # Format choices as "A) choice1\nB) choice2\n..."
            choices_text = example["choices"]["text"]
            choices_labels = example["choices"]["label"]
            
            choices = "\n".join([
                f"{label}) {text}"
                for label, text in zip(choices_labels, choices_text)
            ])
            
            # Insert into template
            prompt = self.preamble.format(
                question=example["question"],
                choices=choices
            )
            prompts.append(prompt)
        
        return prompts
    
    def _tokenize_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompts with padding and truncation.
        
        Args:
            prompts: List of formatted prompt strings
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        return self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len
        )
    
    def collate_fn(self, batch: List[Dict]) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Collate function for DataLoader (causal LM classification mode).
        
        Args:
            batch: List of examples from HuggingFace dataset
        
        Returns:
            Tuple of:
            - prompts: Dict with 'input_ids' and 'attention_mask' [batch_size, seq_len]
            - classes: Tensor of class indices [batch_size] (0-4 for A-E)
            - targets: Tensor of target token IDs [batch_size] (actual token IDs)
        """
        # Format and tokenize prompts
        prompts = self._format_prompts(batch)
        tokenized = self._tokenize_prompts(prompts)
        
        # Extract gold classes
        classes = []
        for example in batch:
            answer_key = example["answerKey"]
            
            # Handle both alphabetical (A, B, C) and numerical (1, 2, 3) answers
            if answer_key.isalpha():
                class_idx = ord(answer_key) - ord("A")
            else:
                class_idx = int(answer_key) - 1
            
            classes.append(class_idx)
        
        classes = torch.tensor(classes, dtype=torch.long)
        
        # Map class indices to target token IDs
        targets = torch.cat([self.label2target[c.item()] for c in classes])
        
        return tokenized, classes, targets
    
    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create DataLoader for a specific split.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
        
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self.dset[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


def load_arc_easy_dataset(
    tokenizer: AutoTokenizer,
    config: Dict[str, Any] = None,
    batch_size: int = None,
    max_seq_len: int = None,
    add_space: bool = None,
    few_shot: bool = None
) -> Dict[str, DataLoader]:
    """
    Convenience function to load ARC-Easy dataloaders.
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary (from YAML)
        batch_size: Batch size for dataloaders (overrides config)
        max_seq_len: Maximum sequence length (overrides config)
        add_space: Whether to add leading space to answer tokens (overrides config)
        few_shot: Whether to use few-shot prompting (overrides config)
    
    Returns:
        Dictionary with 'train', 'validation', 'test' dataloaders and dataset object
    """
    # Get batch_size from config if not provided
    if batch_size is None and config is not None:
        batch_size = config.get('batch_size')
    elif batch_size is None:
        batch_size = 4
    
    dataset = ARCEasyDataset(
        tokenizer=tokenizer,
        config=config,
        add_space=add_space,
        max_seq_len=max_seq_len,
        few_shot=few_shot
    )
    
    dataloaders = {
        "train": dataset.get_dataloader(
            split="train",
            batch_size=batch_size,
            shuffle=True
        ),
        "validation": dataset.get_dataloader(
            split="validation",
            batch_size=batch_size,
            shuffle=False
        ),
        "test": dataset.get_dataloader(
            split="test",
            batch_size=batch_size,
            shuffle=False
        )
    }
    
    return dataloaders, dataset


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("="*60)
    print("Testing ARC-Easy Dataset Loader")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataloaders, dataset = load_arc_easy_dataset(
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_len=512,
        add_space=True
    )
    
    # Test batch
    print("\n" + "="*60)
    print("Sample Batch from Training Set")
    print("="*60)
    
    train_loader = dataloaders["train"]
    batch = next(iter(train_loader))
    inputs, classes, targets = batch
    
    print(f"\nInput IDs shape: {inputs['input_ids'].shape}")
    print(f"Attention mask shape: {inputs['attention_mask'].shape}")
    print(f"Classes: {classes}")
    print(f"Targets: {targets}")
    
    # Decode first example
    print(f"\nFirst prompt (decoded):")
    print(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))
    print(f"\nGold answer: {chr(ord('A') + classes[0].item())}")
    print(f"Target token ID: {targets[0].item()}")

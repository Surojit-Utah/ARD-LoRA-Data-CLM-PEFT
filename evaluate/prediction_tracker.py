"""
Prediction Tracker for ARD-LoRA Training
========================================

This module provides prediction tracking functionality to monitor model learning
progress by saving predictions on fixed examples across training epochs.

Key Features:
1. Selects representative examples from training and validation sets
2. Tracks prediction evolution across epochs
3. Saves human-readable prediction files
4. Provides detailed confidence analysis for multiple choice questions
"""

import os
import json
import random
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset


class PredictionTracker:
    """
    Tracks model predictions on fixed examples across training epochs.
    
    Designed specifically for ARC-Easy multiple choice questions but can be
    extended for other datasets.
    """
    
    def __init__(self, 
                 output_dir: str,
                 tokenizer,
                 n_examples: int = 10,
                 dataset_name: str = "arc_easy",
                 seed: int = 42):
        """
        Initialize prediction tracker.
        
        Args:
            output_dir: Directory to save prediction files
            tokenizer: Tokenizer used by the model
            n_examples: Number of examples to track per split
            dataset_name: Name of dataset for formatting
            seed: Random seed for reproducible example selection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = tokenizer
        self.n_examples = n_examples
        self.dataset_name = dataset_name.lower()
        self.seed = seed
        
        # Fixed examples to track
        self.train_examples = []
        self.val_examples = []
        self.train_indices = []
        self.val_indices = []
        
        # Prediction history
        self.prediction_history = {
            'train': [],
            'val': []
        }
        
        print(f"[PredictionTracker] Initialized with output_dir: {self.output_dir}")
    
    def select_examples(self, train_dataset, val_dataset):
        """
        Select representative examples from training and validation sets.
        
        For ARC-Easy: Ensures we have examples from different answer choices (A/B/C/D).
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print(f"[PredictionTracker] Selecting {self.n_examples} examples from each split...")
        
        # Select training examples
        if train_dataset is not None and len(train_dataset) > 0:
            self.train_indices, self.train_examples = self._select_balanced_examples(
                train_dataset, self.n_examples, "train"
            )
        
        # Select validation examples
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_indices, self.val_examples = self._select_balanced_examples(
                val_dataset, self.n_examples, "val"
            )
        
        # Save selected examples info
        self._save_selected_examples_info()
        
        print(f"[PredictionTracker] Selected {len(self.train_examples)} train + {len(self.val_examples)} val examples")
    
    def _select_balanced_examples(self, dataset, n_examples: int, split_name: str) -> Tuple[List[int], List[Dict]]:
        """
        Select balanced examples ensuring different answer choices are represented.
        
        Args:
            dataset: Dataset to select from
            n_examples: Number of examples to select
            split_name: Name of split for logging
            
        Returns:
            (indices, examples): Selected indices and example data
        """
        total_size = len(dataset)
        
        if self.dataset_name == "arc_easy":
            # For ARC-Easy: Try to get balanced answer choices
            return self._select_arc_easy_balanced(dataset, n_examples, split_name)
        else:
            # For other datasets: Random selection
            indices = random.sample(range(total_size), min(n_examples, total_size))
            examples = []
            
            for idx in indices:
                try:
                    example = dataset[idx]
                    examples.append({
                        'index': idx,
                        'input_ids': example.get('input_ids', []),
                        'labels': example.get('labels', []),
                        'attention_mask': example.get('attention_mask', [])
                    })
                except Exception as e:
                    print(f"[PredictionTracker] Error accessing {split_name}[{idx}]: {e}")
                    continue
            
            return indices, examples
    
    def _select_arc_easy_balanced(self, dataset, n_examples: int, split_name: str) -> Tuple[List[int], List[Dict]]:
        """
        Select ARC-Easy examples with balanced answer choices.
        
        Args:
            dataset: ARC-Easy dataset
            n_examples: Number of examples to select
            split_name: Name of split for logging
            
        Returns:
            (indices, examples): Selected indices and example data
        """
        # Group examples by answer choice
        answer_groups = {'A': [], 'B': [], 'C': [], 'D': []}
        
        # Sample a subset to analyze (for large datasets)
        total_size = len(dataset)
        sample_size = min(1000, total_size)
        sample_indices = random.sample(range(total_size), sample_size)
        
        for idx in sample_indices:
            try:
                example = dataset[idx]
                
                # Extract answer choice from labels
                answer_choice = self._extract_answer_choice_from_example(example)
                
                if answer_choice in answer_groups:
                    answer_groups[answer_choice].append(idx)
                    
            except Exception as e:
                print(f"[PredictionTracker] Error analyzing {split_name}[{idx}]: {e}")
                continue
        
        # Select balanced examples
        examples_per_choice = max(1, n_examples // 4)  # Try to get ~2-3 per choice
        selected_indices = []
        
        for choice, indices in answer_groups.items():
            if indices:
                # Select random examples from this choice
                n_select = min(examples_per_choice, len(indices))
                selected = random.sample(indices, n_select)
                selected_indices.extend(selected)
                print(f"[PredictionTracker] {split_name}: Selected {n_select} examples with answer {choice}")
        
        # If we don't have enough, fill with random examples
        while len(selected_indices) < n_examples and len(selected_indices) < total_size:
            remaining_indices = [i for i in range(total_size) if i not in selected_indices]
            if remaining_indices:
                selected_indices.append(random.choice(remaining_indices))
        
        # Limit to requested number
        selected_indices = selected_indices[:n_examples]
        
        # Get the actual examples
        examples = []
        for idx in selected_indices:
            try:
                example = dataset[idx]
                
                # Decode text for human readability
                input_ids = example.get('input_ids', [])
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True) if input_ids else ""
                
                examples.append({
                    'index': idx,
                    'input_ids': input_ids,
                    'labels': example.get('labels', []),
                    'attention_mask': example.get('attention_mask', []),
                    'text': text,
                    'answer_choice': self._extract_answer_choice_from_example(example)
                })
            except Exception as e:
                print(f"[PredictionTracker] Error processing {split_name}[{idx}]: {e}")
                continue
        
        return selected_indices, examples
    
    def _extract_answer_choice_from_example(self, example) -> str:
        """
        Extract the correct answer choice (A/B/C/D) from an example.
        
        Args:
            example: Dataset example
            
        Returns:
            Answer choice letter or 'Unknown'
        """
        try:
            # Method 1: Check if there's a label field that maps to A/B/C/D
            if 'label' in example:
                label_idx = example['label']
                if isinstance(label_idx, (int, np.integer)) and 0 <= label_idx <= 3:
                    return chr(ord('A') + label_idx)  # 0->A, 1->B, 2->C, 3->D
            
            # Method 2: Look for answer in the text
            input_ids = example.get('input_ids', [])
            if input_ids:
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                
                # Look for "The answer is X" pattern
                answer_match = re.search(r'The answer is ([ABCD])', text)
                if answer_match:
                    return answer_match.group(1)
                
                # Look for other answer patterns
                for pattern in [r'Answer: ([ABCD])', r'Correct answer: ([ABCD])', r'\b([ABCD])\)?\s*$']:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
            
            return 'Unknown'
            
        except Exception:
            return 'Unknown'
    
    def track_predictions(self, model, epoch: int):
        """
        Generate and save predictions for the fixed examples.
        
        Args:
            model: Trained model
            epoch: Current training epoch
        """
        print(f"[PredictionTracker] Generating predictions for epoch {epoch}...")
        
        model.eval()
        
        epoch_predictions = {
            'epoch': epoch,
            'train_predictions': [],
            'val_predictions': []
        }
        
        # Generate predictions for training examples
        if self.train_examples:
            train_preds = self._generate_predictions(model, self.train_examples, "train")
            epoch_predictions['train_predictions'] = train_preds
        
        # Generate predictions for validation examples
        if self.val_examples:
            val_preds = self._generate_predictions(model, self.val_examples, "val")
            epoch_predictions['val_predictions'] = val_preds
        
        # Save predictions to file
        self._save_epoch_predictions(epoch_predictions)
        
        # Store in history
        self.prediction_history['train'].append(train_preds)
        self.prediction_history['val'].append(val_preds)
        
        model.train()  # Restore training mode
        
        print(f"[PredictionTracker] Saved predictions for epoch {epoch}")
    
    def _generate_predictions(self, model, examples: List[Dict], split_name: str) -> List[Dict]:
        """
        Generate predictions for a list of examples.
        
        Args:
            model: Model to use for predictions
            examples: List of examples to predict on
            split_name: Name of split for logging
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i, example in enumerate(examples):
                try:
                    # Prepare input
                    input_ids = torch.tensor([example['input_ids']], device=device)
                    attention_mask = torch.tensor([example.get('attention_mask', [1] * len(example['input_ids']))], device=device)
                    
                    # Generate prediction
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # For ARC-Easy: Extract answer choice predictions
                    prediction_info = self._extract_answer_prediction(
                        example, logits, input_ids, split_name, i
                    )
                    
                    predictions.append(prediction_info)
                    
                except Exception as e:
                    print(f"[PredictionTracker] Error predicting {split_name}[{i}]: {e}")
                    predictions.append({
                        'example_idx': i,
                        'error': str(e),
                        'predicted_answer': 'Error',
                        'confidence': 0.0
                    })
        
        return predictions
    
    def _extract_answer_prediction(self, example: Dict, logits: torch.Tensor, 
                                  input_ids: torch.Tensor, split_name: str, example_idx: int) -> Dict:
        """
        Extract answer choice prediction from model logits.
        
        Args:
            example: Original example data
            logits: Model output logits
            input_ids: Input token IDs
            split_name: Split name for logging
            example_idx: Example index
            
        Returns:
            Prediction information dictionary
        """
        try:
            # Get token probabilities at the last position (answer position)
            last_logits = logits[0, -1, :]  # Last token position
            probs = torch.softmax(last_logits, dim=-1)
            
            # Get answer choice token IDs
            choice_tokens = {}
            for choice in ['A', 'B', 'C', 'D']:
                # Try different formats
                formats = [choice, f' {choice}', f'({choice})', f' ({choice})']
                for fmt in formats:
                    tokens = self.tokenizer.encode(fmt, add_special_tokens=False)
                    if tokens:
                        choice_tokens[choice] = tokens[0]  # Use first token
                        break
            
            # Find the choice with highest probability
            choice_probs = {}
            for choice, token_id in choice_tokens.items():
                if token_id < len(probs):
                    choice_probs[choice] = probs[token_id].item()
                else:
                    choice_probs[choice] = 0.0
            
            # Get prediction
            if choice_probs:
                predicted_choice = max(choice_probs.keys(), key=lambda k: choice_probs[k])
                confidence = choice_probs[predicted_choice]
            else:
                predicted_choice = 'Unknown'
                confidence = 0.0
            
            # Extract correct answer
            correct_answer = example.get('answer_choice', 'Unknown')
            
            # Get human-readable text
            text = example.get('text', '')
            if not text and 'input_ids' in example:
                text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            
            return {
                'example_idx': example_idx,
                'dataset_idx': example.get('index', -1),
                'text': text[:500] + '...' if len(text) > 500 else text,  # Truncate for readability
                'correct_answer': correct_answer,
                'predicted_answer': predicted_choice,
                'confidence': confidence,
                'choice_probabilities': choice_probs,
                'is_correct': predicted_choice == correct_answer,
                'choice_tokens': choice_tokens
            }
            
        except Exception as e:
            return {
                'example_idx': example_idx,
                'dataset_idx': example.get('index', -1),
                'error': str(e),
                'predicted_answer': 'Error',
                'confidence': 0.0,
                'is_correct': False
            }
    
    def _save_epoch_predictions(self, epoch_predictions: Dict):
        """
        Save predictions for an epoch to human-readable text file.
        
        Args:
            epoch_predictions: Prediction data for the epoch
        """
        epoch = epoch_predictions['epoch']
        
        # Create predictions file
        pred_file = self.output_dir / f"predictions_epoch_{epoch:03d}.txt"
        
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write(f"PREDICTION TRACKING - EPOCH {epoch}\n")
            f.write("=" * 80 + "\n\n")
            
            # Training predictions
            if epoch_predictions['train_predictions']:
                f.write(f"TRAINING EXAMPLES ({len(epoch_predictions['train_predictions'])} samples)\n")
                f.write("-" * 50 + "\n")
                
                for pred in epoch_predictions['train_predictions']:
                    self._write_prediction_to_file(f, pred)
                
                f.write("\n\n")
            
            # Validation predictions
            if epoch_predictions['val_predictions']:
                f.write(f"VALIDATION EXAMPLES ({len(epoch_predictions['val_predictions'])} samples)\n")
                f.write("-" * 50 + "\n")
                
                for pred in epoch_predictions['val_predictions']:
                    self._write_prediction_to_file(f, pred)
        
        # Also save as JSON for programmatic analysis
        json_file = self.output_dir / f"predictions_epoch_{epoch:03d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_predictions, f, indent=2, ensure_ascii=False)
    
    def _write_prediction_to_file(self, f, prediction: Dict):
        """
        Write a single prediction to file in human-readable format.
        
        Args:
            f: File handle
            prediction: Prediction dictionary
        """
        if 'error' in prediction:
            f.write(f"Example {prediction['example_idx']}: ERROR - {prediction['error']}\n\n")
            return
        
        f.write(f"Example {prediction['example_idx']} (Dataset Index: {prediction.get('dataset_idx', 'N/A')})\n")
        f.write(f"{'='*60}\n")
        
        # Question text
        f.write("QUESTION:\n")
        f.write(f"{prediction.get('text', 'N/A')}\n\n")
        
        # Prediction results
        correct = prediction.get('correct_answer', 'Unknown')
        predicted = prediction.get('predicted_answer', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        is_correct = prediction.get('is_correct', False)
        
        f.write("RESULTS:\n")
        f.write(f"Correct Answer:   {correct}\n")
        f.write(f"Predicted Answer: {predicted}\n")
        f.write(f"Confidence:       {confidence:.4f}\n")
        f.write(f"Result:           {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")
        
        # Choice probabilities
        if 'choice_probabilities' in prediction and prediction['choice_probabilities']:
            f.write("\nCHOICE PROBABILITIES:\n")
            for choice in ['A', 'B', 'C', 'D']:
                prob = prediction['choice_probabilities'].get(choice, 0.0)
                marker = " ←" if choice == predicted else ""
                f.write(f"  {choice}: {prob:.4f}{marker}\n")
        
        f.write("\n" + "-"*60 + "\n\n")
    
    def _save_selected_examples_info(self):
        """Save information about selected examples for reference."""
        info_file = self.output_dir / "selected_examples_info.txt"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("SELECTED EXAMPLES FOR PREDICTION TRACKING\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Random Seed: {self.seed}\n")
            f.write(f"Examples per split: {self.n_examples}\n\n")
            
            # Training examples info
            if self.train_examples:
                f.write(f"TRAINING EXAMPLES ({len(self.train_examples)} selected)\n")
                f.write("-" * 50 + "\n")
                
                for i, example in enumerate(self.train_examples):
                    f.write(f"Example {i} (Dataset Index: {example.get('index', 'N/A')})\n")
                    f.write(f"  Answer: {example.get('answer_choice', 'Unknown')}\n")
                    text = example.get('text', '')[:200]
                    f.write(f"  Text: {text}{'...' if len(example.get('text', '')) > 200 else ''}\n\n")
            
            # Validation examples info
            if self.val_examples:
                f.write(f"VALIDATION EXAMPLES ({len(self.val_examples)} selected)\n")
                f.write("-" * 50 + "\n")
                
                for i, example in enumerate(self.val_examples):
                    f.write(f"Example {i} (Dataset Index: {example.get('index', 'N/A')})\n")
                    f.write(f"  Answer: {example.get('answer_choice', 'Unknown')}\n")
                    text = example.get('text', '')[:200]
                    f.write(f"  Text: {text}{'...' if len(example.get('text', '')) > 200 else ''}\n\n")
        
        print(f"[PredictionTracker] Saved example selection info to: {info_file}")
    
    def generate_progress_summary(self):
        """Generate a summary of prediction progress across all epochs."""
        if not self.prediction_history['train'] and not self.prediction_history['val']:
            return
        
        summary_file = self.output_dir / "prediction_progress_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PREDICTION PROGRESS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Calculate accuracy trends
            train_accuracies = []
            val_accuracies = []
            
            for epoch_preds in self.prediction_history['train']:
                if epoch_preds:
                    correct = sum(1 for p in epoch_preds if p.get('is_correct', False))
                    total = len(epoch_preds)
                    train_accuracies.append(correct / total if total > 0 else 0.0)
            
            for epoch_preds in self.prediction_history['val']:
                if epoch_preds:
                    correct = sum(1 for p in epoch_preds if p.get('is_correct', False))
                    total = len(epoch_preds)
                    val_accuracies.append(correct / total if total > 0 else 0.0)
            
            # Write summary
            f.write("ACCURACY EVOLUTION:\n")
            f.write("-" * 30 + "\n")
            
            for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
                f.write(f"Epoch {epoch:2d}: Train={train_acc:.3f}, Val={val_acc:.3f}\n")
            
            if train_accuracies:
                f.write(f"\nTrain Accuracy: {train_accuracies[0]:.3f} → {train_accuracies[-1]:.3f} "
                       f"(Δ={train_accuracies[-1] - train_accuracies[0]:+.3f})\n")
            
            if val_accuracies:
                f.write(f"Val Accuracy:   {val_accuracies[0]:.3f} → {val_accuracies[-1]:.3f} "
                       f"(Δ={val_accuracies[-1] - val_accuracies[0]:+.3f})\n")
        
        print(f"[PredictionTracker] Generated progress summary: {summary_file}")
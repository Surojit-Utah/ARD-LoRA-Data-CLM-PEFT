"""
Dataset Sample Loader and Analyzer
==================================

This script loads actual data samples from the configured datasets and provides
hands-on exploration of the data structure, preprocessing, and task formats.

Usage:
    python dataset_sample_loader.py

Features:
    - Load actual dataset samples
    - Show data preprocessing pipeline
    - Demonstrate tokenization and formatting
    - Compare task formats across datasets
    - Self-contained with error handling
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset, Dataset
    import torch
    from transformers import AutoTokenizer
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

def check_dependencies():
    """Check if required dependencies are available."""
    if not HAS_DEPENDENCIES:
        print("❌ Missing dependencies. Please install:")
        print("   pip install datasets transformers torch")
        print("   Then re-run this script.")
        return False
    return True

def get_dataset_configs():
    """
    Dataset configurations matching the YAML file.
    Returns dataset loading configurations.
    """
    return {
        # Commonsense Reasoning
        "piqa": {
            "path": "piqa",
            "config": None,
            "text_fields": ["goal", "sol1", "sol2"],
            "label_field": "label",
            "format_template": "Goal: {goal}\nOption A: {sol1}\nOption B: {sol2}\nAnswer:"
        },
        
        "hellaswag": {
            "path": "hellaswag", 
            "config": None,
            "text_fields": ["ctx", "endings"],
            "label_field": "label",
            "format_template": "Context: {ctx}\nOption A: {ending0}\nOption B: {ending1}\nOption C: {ending2}\nOption D: {ending3}\nAnswer:"
        },
        
        "winogrande": {
            "path": "winogrande",
            "config": "winogrande_xl",  # We'll use xl to show both s and m subsets
            "text_fields": ["sentence", "option1", "option2"],
            "label_field": "answer",
            "format_template": "Sentence: {sentence}\nOption A: {option1}\nOption B: {option2}\nAnswer:"
        },
        
        # Science QA
        "arc_easy": {
            "path": "ai2_arc",
            "config": "ARC-Easy",
            "text_fields": ["question", "choices"],
            "label_field": "answerKey",
            "format_template": "Question: {question}\nA: {choice_A}\nB: {choice_B}\nC: {choice_C}\nD: {choice_D}\nAnswer:"
        },
        
        "arc_challenge": {
            "path": "ai2_arc",
            "config": "ARC-Challenge", 
            "text_fields": ["question", "choices"],
            "label_field": "answerKey",
            "format_template": "Question: {question}\nA: {choice_A}\nB: {choice_B}\nC: {choice_C}\nD: {choice_D}\nAnswer:"
        },
        
        # Reading Comprehension
        "boolq": {
            "path": "boolq",
            "config": None,
            "text_fields": ["passage", "question"],
            "label_field": "answer",
            "format_template": "Passage: {passage}\nQuestion: {question}\nAnswer:"
        },
        
        "openbookqa": {
            "path": "openbookqa",
            "config": "main",
            "text_fields": ["question_stem", "choices"],
            "label_field": "answerKey",
            "format_template": "Question: {question_stem}\nA: {choice_A}\nB: {choice_B}\nC: {choice_C}\nD: {choice_D}\nAnswer:"
        },
        
        # Natural Language Inference
        "anli": {
            "path": "anli",
            "config": None,
            "text_fields": ["premise", "hypothesis"],
            "label_field": "label",
            "format_template": "Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
        },
        
        "rte": {
            "path": "super_glue",
            "config": "rte",
            "text_fields": ["premise", "hypothesis"],
            "label_field": "label",
            "format_template": "Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
        },
        
        "cb": {
            "path": "super_glue",
            "config": "cb",
            "text_fields": ["premise", "hypothesis"],
            "label_field": "label", 
            "format_template": "Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
        },
        
        "copa": {
            "path": "super_glue",
            "config": "copa",
            "text_fields": ["premise", "choice1", "choice2"],
            "label_field": "label",
            "format_template": "Premise: {premise}\nChoice A: {choice1}\nChoice B: {choice2}\nAnswer:"
        },
        
        # Sentiment Analysis
        "sst2": {
            "path": "glue",
            "config": "sst2",
            "text_fields": ["sentence"],
            "label_field": "label",
            "format_template": "Sentence: {sentence}\nSentiment:"
        }
    }

def load_dataset_sample(dataset_name: str, config: Dict[str, Any], num_samples: int = 3) -> Optional[Dataset]:
    """Load a small sample from a dataset."""
    try:
        print(f"🔄 Loading {dataset_name}...")
        
        if config["config"]:
            dataset = load_dataset(config["path"], config["config"], split="train")
        else:
            dataset = load_dataset(config["path"], split="train")
        
        # Get a small sample
        sample_size = min(num_samples, len(dataset))
        sample_dataset = dataset.select(range(sample_size))
        
        print(f"✅ Loaded {sample_size} samples from {dataset_name}")
        return sample_dataset
        
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {str(e)}")
        return None

def format_example(example: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Format a single example according to the template."""
    try:
        template = config["format_template"]
        
        # Handle different dataset formats
        if "choices" in example and isinstance(example["choices"], dict):
            # Handle ARC format
            choices = example["choices"]
            if "text" in choices:
                for i, choice in enumerate(choices["text"]):
                    template = template.replace(f"{{choice_{choices['label'][i]}}}", choice)
        
        elif "choices" in example and isinstance(example["choices"], list):
            # Handle OpenBookQA format
            choices = example["choices"]
            if len(choices) >= 4:
                template = template.replace("{choice_A}", choices[0]["text"] if isinstance(choices[0], dict) else str(choices[0]))
                template = template.replace("{choice_B}", choices[1]["text"] if isinstance(choices[1], dict) else str(choices[1]))
                template = template.replace("{choice_C}", choices[2]["text"] if isinstance(choices[2], dict) else str(choices[2]))
                template = template.replace("{choice_D}", choices[3]["text"] if isinstance(choices[3], dict) else str(choices[3]))
        
        elif "endings" in example:
            # Handle HellaSwag format
            endings = example["endings"]
            for i, ending in enumerate(endings):
                template = template.replace(f"{{ending{i}}}", ending)
        
        # Replace standard fields
        for field in config["text_fields"]:
            if field in example:
                template = template.replace(f"{{{field}}}", str(example[field]))
        
        return template
        
    except Exception as e:
        return f"Error formatting example: {str(e)}"

def analyze_dataset_sample(dataset_name: str, sample_dataset: Dataset, config: Dict[str, Any]):
    """Analyze and display information about a dataset sample."""
    print(f"\n🔬 ANALYZING: {dataset_name.upper()}")
    print("=" * 60)
    
    if sample_dataset is None or len(sample_dataset) == 0:
        print("❌ No data available for analysis")
        return
    
    # Basic statistics
    print(f"📊 Sample Size: {len(sample_dataset)}")
    print(f"🏷️  Features: {list(sample_dataset.features.keys())}")
    
    # Show label distribution
    label_field = config.get("label_field")
    if label_field and label_field in sample_dataset.features:
        labels = [example[label_field] for example in sample_dataset]
        unique_labels = list(set(labels))
        print(f"🎯 Unique Labels: {unique_labels}")
        print(f"📈 Label Distribution: {dict(zip(unique_labels, [labels.count(l) for l in unique_labels]))}")
    
    print("\n📝 SAMPLE EXAMPLES:")
    print("-" * 40)
    
    # Show formatted examples
    for i, example in enumerate(sample_dataset):
        print(f"\nExample {i+1}:")
        print("-" * 20)
        
        # Show raw data structure
        print("Raw data:")
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
        
        print("\nFormatted:")
        formatted = format_example(example, config)
        print(f"  {formatted}")
        
        # Show expected answer
        if label_field and label_field in example:
            label_value = example[label_field]
            print(f"  Expected: {label_value}")

def demonstrate_tokenization(text: str, dataset_name: str):
    """Demonstrate tokenization with LLaMA tokenizer."""
    try:
        # Use a small model for demonstration
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        print(f"\n🔤 TOKENIZATION DEMO ({dataset_name})")
        print("-" * 40)
        print(f"Original text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"First 10 tokens: {tokens[:10]}")
        print(f"Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        
    except Exception as e:
        print(f"❌ Tokenization demo failed: {str(e)}")

def compare_task_formats(configs: Dict[str, Dict[str, Any]]):
    """Compare task formats across different datasets."""
    print("\n🔄 TASK FORMAT COMPARISON")
    print("=" * 60)
    
    format_types = {}
    for dataset_name, config in configs.items():
        template = config["format_template"]
        # Categorize by format type
        if "Option A:" in template and "Option B:" in template:
            if "Option C:" in template:
                format_type = "Multiple Choice (4 options)"
            else:
                format_type = "Multiple Choice (2 options)"
        elif "Choice A:" in template:
            format_type = "Binary Choice"
        elif "Label:" in template:
            format_type = "Classification"
        elif "Sentiment:" in template:
            format_type = "Sentiment"
        else:
            format_type = "Other"
        
        if format_type not in format_types:
            format_types[format_type] = []
        format_types[format_type].append(dataset_name)
    
    for format_type, datasets in format_types.items():
        print(f"\n📋 {format_type}:")
        for dataset in datasets:
            print(f"   • {dataset}")

def print_ard_lora_preprocessing_notes():
    """Print notes about preprocessing for ARD-LoRA."""
    print("\n🔧 ARD-LORA PREPROCESSING NOTES")
    print("=" * 60)
    print("Key considerations when preprocessing these datasets for ARD-LoRA:")
    print()
    print("🎯 Label Encoding:")
    print("   • Binary tasks: 0/1 encoding")
    print("   • Multi-class: 0/1/2/3 encoding")
    print("   • Text labels converted to integers")
    print()
    print("📏 Sequence Length:")
    print("   • Target max length: 512 tokens")
    print("   • Truncation strategy: from the left for context")
    print("   • Padding: to max length in batch")
    print()
    print("🔤 Tokenization:")
    print("   • LLaMA tokenizer with special tokens")
    print("   • Add EOS token at the end")
    print("   • Handle unknown tokens gracefully")
    print()
    print("🎲 ARD-Specific:")
    print("   • Need consistent batch sizes for variance estimation")
    print("   • Multiple forward passes for uncertainty quantification")
    print("   • Gradient tracking for relevance determination")

def main():
    """Main function to run the dataset sample loader."""
    print("🔍 DATASET SAMPLE LOADER AND ANALYZER")
    print("=" * 80)
    print("This script loads actual samples from configured datasets and")
    print("demonstrates data preprocessing for ARD-LoRA training.")
    print("=" * 80)
    
    if not check_dependencies():
        return
    
    configs = get_dataset_configs()
    
    # Interactive dataset selection
    print("\n📋 Available datasets:")
    dataset_list = list(configs.keys())
    for i, dataset_name in enumerate(dataset_list, 1):
        print(f"{i:2d}. {dataset_name}")
    
    print("\nChoose datasets to analyze:")
    print("Enter numbers separated by spaces (e.g., '1 3 5') or 'all' for all datasets")
    
    try:
        choice = input("Your choice: ").strip().lower()
        
        if choice == 'all':
            selected_datasets = dataset_list
        else:
            indices = [int(x) - 1 for x in choice.split()]
            selected_datasets = [dataset_list[i] for i in indices if 0 <= i < len(dataset_list)]
        
        if not selected_datasets:
            print("No valid datasets selected.")
            return
        
        print(f"\n🎯 Analyzing {len(selected_datasets)} dataset(s)...")
        
        # Analyze each selected dataset
        for dataset_name in selected_datasets:
            config = configs[dataset_name]
            
            # Load sample
            sample_dataset = load_dataset_sample(dataset_name, config, num_samples=2)
            
            # Analyze
            analyze_dataset_sample(dataset_name, sample_dataset, config)
            
            # Demonstrate tokenization on first example
            if sample_dataset and len(sample_dataset) > 0:
                first_example = sample_dataset[0]
                formatted_text = format_example(first_example, config)
                demonstrate_tokenization(formatted_text, dataset_name)
            
            print("\n" + "="*80)
        
        # Show comparisons
        compare_task_formats(configs)
        print_ard_lora_preprocessing_notes()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("You now have hands-on experience with the actual data formats")
    print("and preprocessing requirements for ARD-LoRA training.")
    print("=" * 80)

if __name__ == "__main__":
    main()
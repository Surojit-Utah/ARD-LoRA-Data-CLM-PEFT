"""
ARC-Easy Label Masking Validation Script
========================================

This script tests the ARC-Easy specific answer masking implementation
in the dataloader to ensure it correctly identifies and unmasks only
the final answer tokens (A, B, C, D) while masking everything else.

Usage:
    python scripts/test_arc_easy_masking.py
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from dataloader.bayesian_peft_cached import load_bayesian_peft_with_caching


def test_arc_easy_masking():
    """Test the ARC-Easy masking implementation with actual data"""
    
    print("=" * 80)
    print("ARC-EASY LABEL MASKING VALIDATION")
    print("=" * 80)
    
    # Use configuration from existing setup
    config = CONFIG or {}
    
    # Merge config properly (like in run_training_cached.py)
    if "datasets" in config and "arc_easy" in config["datasets"]:
        dataset_config = config["datasets"]["arc_easy"]
        config.update(dataset_config)
    
    # Set defaults for testing
    config["dataset_name_specific"] = "arc_easy"
    config["max_len"] = config.get("max_len", 512)  # Shorter for testing
    
    # Use cached data directory
    cache_root = "G:/My Drive/ARD_LoRA_Data_Cache"  # Use existing cache
    
    print(f"[CONFIG] Dataset: {config['dataset_name_specific']}")
    print(f"[CONFIG] Max length: {config['max_len']}")
    print(f"[CONFIG] Cache root: {cache_root}")
    
    try:
        # Load tokenizer from config with fallback to public tokenizers
        tokenizer_name = (
            config.get("tokenizer_name") or 
            config.get("model_name_or_path") or
            config.get("model_name") or
            "gpt2"  # Default to public tokenizer for testing
        )
        
        print(f"[INFO] Attempting to use tokenizer: {tokenizer_name}")
        
        # Try the configured tokenizer first, then fallback to public ones
        from transformers import AutoTokenizer
        tokenizer = None
        tokenizer_attempts = [tokenizer_name, "gpt2", "microsoft/DialoGPT-medium", "distilgpt2"]
        
        for attempt_name in tokenizer_attempts:
            try:
                tokenizer = AutoTokenizer.from_pretrained(attempt_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"[SUCCESS] Loaded tokenizer: {attempt_name}")
                tokenizer_name = attempt_name  # Update for dataset loading
                break
            except Exception as e:
                print(f"[FAILED] {attempt_name}: {str(e)[:100]}...")
                continue
        
        if tokenizer is None:
            print("[ERROR] Could not load any tokenizer")
            return
        
        print(f"[INFO] Loading datasets with tokenizer: {tokenizer_name}")
        
        # Load datasets using the actual dataloader
        train_ds, val_ds, tokenizer = load_bayesian_peft_with_caching(
            dataset_name="arc_easy",
            tokenizer_name=tokenizer_name,
            config=config,
            cache_root=cache_root
        )
        
        if not train_ds or len(train_ds) == 0:
            print("[ERROR] No training data loaded")
            return
        
        print(f"[SUCCESS] Loaded datasets - Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}")
        print(f"[INFO] Tokenizer PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"[INFO] Tokenizer EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
        # Test on first few examples
        num_test_samples = min(5, len(train_ds))
        print(f"\n[TESTING] Analyzing first {num_test_samples} samples for masking behavior:")
        
        for i in range(num_test_samples):
            print(f"\n--- SAMPLE {i+1} ---")
            
            example = train_ds[i]
            input_ids = example["input_ids"]
            labels = example["labels"]
            attention_mask = example.get("attention_mask", [1] * len(input_ids))
            
            # Find actual content (non-padding tokens)
            pad_token_id = tokenizer.pad_token_id
            actual_content_length = 0
            for j, token_id in enumerate(input_ids):
                if token_id != pad_token_id:
                    actual_content_length = j + 1
            
            # Decode the actual content (non-padding)
            actual_input_ids = input_ids[:actual_content_length]
            actual_labels = labels[:actual_content_length]
            
            full_text = tokenizer.decode(actual_input_ids, skip_special_tokens=True)
            print(f"Full text: {full_text}")
            print(f"Actual content length: {actual_content_length} / {len(input_ids)} tokens")
            
            # Analyze masking on actual content only
            total_actual_tokens = len(actual_labels)
            unmasked_positions = [j for j, label in enumerate(actual_labels) if label != -100]
            masked_positions = [j for j, label in enumerate(actual_labels) if label == -100]
            
            print(f"Actual tokens: {total_actual_tokens}")
            print(f"Masked tokens: {len(masked_positions)}")
            print(f"Unmasked tokens: {len(unmasked_positions)}")
            
            if unmasked_positions:
                # Show unmasked tokens from actual content
                unmasked_token_ids = [actual_input_ids[j] for j in unmasked_positions]
                unmasked_text = tokenizer.decode(unmasked_token_ids, skip_special_tokens=True)
                print(f"Unmasked text: '{unmasked_text}'")
                print(f"Unmasked positions: {unmasked_positions}")
                
                # Show context around unmasked tokens (only first few)
                for idx, pos in enumerate(unmasked_positions[:5]):  # Only show first 5
                    token_text = tokenizer.decode([actual_input_ids[pos]])
                    context_start = max(0, pos - 3)
                    context_end = min(len(actual_input_ids), pos + 4)
                    context_tokens = actual_input_ids[context_start:context_end]
                    context_text = tokenizer.decode(context_tokens)
                    print(f"  Position {pos}: '{token_text}' in context: '{context_text}'")
                
                # Check if this looks like proper answer masking
                if len(unmasked_positions) <= 3:  # Should be 1-2 tokens for A/B/C/D
                    last_unmasked = max(unmasked_positions)
                    
                    if last_unmasked >= actual_content_length * 0.8:  # Answer should be near the end
                        print(f"  ✓ GOOD: Answer appears near sequence end (pos {last_unmasked}/{actual_content_length})")
                    else:
                        print(f"  ⚠ WARNING: Answer not near end (pos {last_unmasked}/{actual_content_length})")
                        
                    if unmasked_text.strip() in ['A', 'B', 'C', 'D']:
                        print(f"  ✓ GOOD: Unmasked text is a valid answer choice")
                    else:
                        print(f"  ⚠ WARNING: Unmasked text '{unmasked_text}' doesn't look like A/B/C/D")
                else:
                    print(f"  ⚠ WARNING: Too many unmasked tokens ({len(unmasked_positions)}) - expected 1-2")
            else:
                print(f"  ❌ ERROR: No unmasked tokens found!")
                
            # Show detailed token-by-token breakdown for first sample (actual content only)
            if i == 0:
                print(f"\nDetailed token breakdown (actual content only):")
                for j, (token_id, label) in enumerate(zip(actual_input_ids, actual_labels)):
                    token_text = tokenizer.decode([token_id])
                    status = "UNMASKED" if label != -100 else "MASKED"
                    print(f"  {j:2d}: {token_id:5d} -> '{token_text:10s}' ({status})")
            
            # Debug: Check if we can find A/B/C/D patterns manually
            if i == 0:
                print(f"\nDEBUG: Manual search for A/B/C/D patterns:")
                for choice in ['A', 'B', 'C', 'D']:
                    patterns = [
                        tokenizer.encode(choice, add_special_tokens=False),
                        tokenizer.encode(f" {choice}", add_special_tokens=False),
                        tokenizer.encode(f"({choice})", add_special_tokens=False),
                        tokenizer.encode(f" ({choice})", add_special_tokens=False),
                    ]
                    
                    for pattern in patterns:
                        if pattern:  # Only check non-empty patterns
                            for start_pos in range(len(actual_input_ids) - len(pattern) + 1):
                                if actual_input_ids[start_pos:start_pos + len(pattern)] == pattern:
                                    found_text = tokenizer.decode(pattern)
                                    print(f"  Found '{choice}' pattern '{found_text}' at position {start_pos}")
                    
                # Also try searching from the end
                print(f"\nDEBUG: Searching from sequence end (last 20 tokens):")
                search_start = max(0, actual_content_length - 20)
                end_tokens = actual_input_ids[search_start:]
                end_text = tokenizer.decode(end_tokens)
                print(f"  Last 20 tokens: '{end_text}'")
                
                for j, token_id in enumerate(end_tokens):
                    token_text = tokenizer.decode([token_id])
                    actual_pos = search_start + j
                    print(f"    {actual_pos:2d}: {token_id:5d} -> '{token_text}'")
        
        # Summary
        print(f"\n" + "=" * 80)
        print("MASKING VALIDATION SUMMARY")
        print("=" * 80)
        
        total_samples_tested = num_test_samples
        samples_with_unmasked = 0
        samples_with_good_answers = 0
        
        for i in range(num_test_samples):
            example = train_ds[i]
            labels = example["labels"]
            unmasked_positions = [j for j, label in enumerate(labels) if label != -100]
            
            if unmasked_positions:
                samples_with_unmasked += 1
                unmasked_token_ids = [example["input_ids"][j] for j in unmasked_positions]
                unmasked_text = tokenizer.decode(unmasked_token_ids, skip_special_tokens=True).strip()
                
                if unmasked_text in ['A', 'B', 'C', 'D']:
                    samples_with_good_answers += 1
        
        print(f"Samples tested: {total_samples_tested}")
        print(f"Samples with unmasked tokens: {samples_with_unmasked}")
        print(f"Samples with valid answer tokens: {samples_with_good_answers}")
        print(f"Success rate: {samples_with_good_answers/total_samples_tested*100:.1f}%")
        
        if samples_with_good_answers == total_samples_tested:
            print("✅ ALL TESTS PASSED: Masking appears to be working correctly")
        elif samples_with_good_answers > 0:
            print("⚠️ PARTIAL SUCCESS: Some samples have correct masking")
        else:
            print("❌ TESTS FAILED: No samples have correct answer masking")
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_arc_easy_masking()
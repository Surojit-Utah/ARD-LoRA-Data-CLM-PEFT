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
        # Load tokenizer from config with fallback
        tokenizer_name = (
            config.get("tokenizer_name") or 
            config.get("model_name_or_path") or
            config.get("model_name") or
            "meta-llama/Llama-2-7b-hf"  # Default fallback
        )
        
        print(f"[INFO] Using tokenizer: {tokenizer_name}")
        
        if not tokenizer_name:
            print("[ERROR] No tokenizer name found in config")
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
            
            # Decode the full sequence
            full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Full text: {full_text}")
            
            # Analyze masking
            total_tokens = len(labels)
            unmasked_positions = [j for j, label in enumerate(labels) if label != -100]
            masked_positions = [j for j, label in enumerate(labels) if label == -100]
            
            print(f"Total tokens: {total_tokens}")
            print(f"Masked tokens: {len(masked_positions)}")
            print(f"Unmasked tokens: {len(unmasked_positions)}")
            
            if unmasked_positions:
                # Show unmasked tokens
                unmasked_token_ids = [input_ids[j] for j in unmasked_positions]
                unmasked_text = tokenizer.decode(unmasked_token_ids, skip_special_tokens=True)
                print(f"Unmasked text: '{unmasked_text}'")
                print(f"Unmasked positions: {unmasked_positions}")
                
                # Show context around unmasked tokens
                for pos in unmasked_positions:
                    token_text = tokenizer.decode([input_ids[pos]])
                    context_start = max(0, pos - 3)
                    context_end = min(len(input_ids), pos + 4)
                    context_tokens = input_ids[context_start:context_end]
                    context_text = tokenizer.decode(context_tokens)
                    print(f"  Position {pos}: '{token_text}' in context: '{context_text}'")
                
                # Check if this looks like proper answer masking
                if len(unmasked_positions) <= 3:  # Should be 1-2 tokens for A/B/C/D
                    last_unmasked = max(unmasked_positions)
                    total_len = len([t for t in input_ids if t != tokenizer.pad_token_id])
                    
                    if last_unmasked >= total_len * 0.8:  # Answer should be near the end
                        print(f"  ✓ GOOD: Answer appears near sequence end (pos {last_unmasked}/{total_len})")
                    else:
                        print(f"  ⚠ WARNING: Answer not near end (pos {last_unmasked}/{total_len})")
                        
                    if unmasked_text.strip() in ['A', 'B', 'C', 'D']:
                        print(f"  ✓ GOOD: Unmasked text is a valid answer choice")
                    else:
                        print(f"  ⚠ WARNING: Unmasked text '{unmasked_text}' doesn't look like A/B/C/D")
                else:
                    print(f"  ⚠ WARNING: Too many unmasked tokens ({len(unmasked_positions)}) - expected 1-2")
            else:
                print(f"  ❌ ERROR: No unmasked tokens found!")
                
            # Show detailed token-by-token breakdown for first sample
            if i == 0:
                print(f"\nDetailed token breakdown:")
                for j, (token_id, label) in enumerate(zip(input_ids, labels)):
                    if token_id == tokenizer.pad_token_id:
                        continue  # Skip padding tokens
                    token_text = tokenizer.decode([token_id])
                    status = "UNMASKED" if label != -100 else "MASKED"
                    print(f"  {j:2d}: {token_id:5d} -> '{token_text:10s}' ({status})")
        
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
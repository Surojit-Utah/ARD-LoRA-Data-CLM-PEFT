"""
Dataset Explorer for ARD-LoRA Training
======================================

This script provides an interactive walkthrough of all datasets defined in the 
YAML configuration file. It helps understand the tasks, dataset sizes, and 
characteristics of each dataset used for ARD-LoRA training.

Usage:
    python dataset_explorer.py

Features:
    - Interactive exploration of all configured datasets
    - Task descriptions and dataset statistics
    - Sample data preview
    - Performance benchmarks context
    - Self-contained with all dependencies
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def get_dataset_info():
    """
    Comprehensive information about each dataset configured in the YAML file.
    Returns a dictionary with detailed information about each dataset.
    """
    return {
        # Commonsense Reasoning
        "piqa": {
            "name": "Physical Interaction: Question Answering",
            "task_type": "Multiple Choice (2 options)",
            "domain": "Commonsense Reasoning",
            "description": "Questions about physical commonsense in everyday situations",
            "example": "Q: To separate egg whites from the yolk using a water bottle, you should... A) squeeze the bottle, then place it over the yolk B) place the bottle over the yolk, then squeeze",
            "num_labels": 2,
            "typical_size": {"train": 16113, "validation": 1838},
            "evaluation_metric": "Accuracy",
            "difficulty": "Medium",
            "why_important": "Tests physical reasoning and practical knowledge"
        },
        
        "hellaswag": {
            "name": "HellaSwag",
            "task_type": "Multiple Choice (4 options)", 
            "domain": "Commonsense Reasoning",
            "description": "Commonsense inference about everyday activities and situations",
            "example": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid getting a bath. She...",
            "num_labels": 4,
            "typical_size": {"train": 39905, "validation": 10042},
            "evaluation_metric": "Accuracy",
            "difficulty": "Hard",
            "why_important": "Challenging benchmark for commonsense reasoning in complex scenarios"
        },
        
        "winogrande_s": {
            "name": "WinoGrande Small",
            "task_type": "Binary Choice",
            "domain": "Commonsense Reasoning", 
            "description": "Pronoun resolution requiring commonsense reasoning (small subset)",
            "example": "The trophy doesn't fit in the brown suitcase because _ is too large. (the trophy/the suitcase)",
            "num_labels": 2,
            "typical_size": {"train": 640, "validation": 162},
            "evaluation_metric": "Accuracy",
            "difficulty": "Hard",
            "why_important": "Tests pronoun resolution and implicit reasoning with smaller dataset for faster training"
        },
        
        "winogrande_m": {
            "name": "WinoGrande Medium", 
            "task_type": "Binary Choice",
            "domain": "Commonsense Reasoning",
            "description": "Pronoun resolution requiring commonsense reasoning (medium subset)",
            "example": "The trophy doesn't fit in the brown suitcase because _ is too large. (the trophy/the suitcase)",
            "num_labels": 2,
            "typical_size": {"train": 2558, "validation": 1267},
            "evaluation_metric": "Accuracy", 
            "difficulty": "Hard",
            "why_important": "Tests pronoun resolution and implicit reasoning with medium-sized dataset"
        },
        
        # Reading Comprehension
        "arc_easy": {
            "name": "AI2 Reasoning Challenge Easy",
            "task_type": "Multiple Choice (4 options)",
            "domain": "Science QA",
            "description": "Grade-school level science questions (easier subset)",
            "example": "Q: Which tool is used to measure how hot or cold something is? A) Ruler B) Scale C) Thermometer D) Telescope",
            "num_labels": 4,
            "typical_size": {"train": 2251, "validation": 570},
            "evaluation_metric": "Accuracy",
            "difficulty": "Easy",
            "why_important": "Tests basic scientific reasoning and factual knowledge"
        },
        
        "arc_challenge": {
            "name": "AI2 Reasoning Challenge",
            "task_type": "Multiple Choice (4 options)",
            "domain": "Science QA", 
            "description": "Grade-school level science questions (challenging subset)",
            "example": "Q: A student is trying to identify a mineral that scratches glass but is scratched by a steel file. What property is being tested? A) Hardness B) Luster C) Density D) Cleavage",
            "num_labels": 4,
            "typical_size": {"train": 1119, "validation": 299},
            "evaluation_metric": "Accuracy",
            "difficulty": "Hard",
            "why_important": "Tests advanced scientific reasoning and complex problem solving"
        },
        
        "boolq": {
            "name": "BoolQ",
            "task_type": "Yes/No Questions",
            "domain": "Reading Comprehension",
            "description": "Yes/No questions about Wikipedia passages",
            "example": "Passage: [about cats] Question: Can cats see in complete darkness? Answer: No",
            "num_labels": 2,
            "typical_size": {"train": 9427, "validation": 3270},
            "evaluation_metric": "Accuracy",
            "difficulty": "Medium",
            "why_important": "Tests reading comprehension and factual reasoning"
        },
        
        "openbookqa": {
            "name": "OpenBookQA",
            "task_type": "Multiple Choice (4 options)",
            "domain": "Science QA",
            "description": "Elementary-level science questions requiring multi-step reasoning",
            "example": "Q: A pencil falls off of a desk. What makes the pencil fall? A) Gravity B) Magnetism C) Electricity D) Friction",
            "num_labels": 4,
            "typical_size": {"train": 4957, "validation": 500},
            "evaluation_metric": "Accuracy",
            "difficulty": "Medium",
            "why_important": "Tests multi-hop reasoning and scientific knowledge application"
        },
        
        # Natural Language Inference
        "anli": {
            "name": "Adversarial NLI",
            "task_type": "3-way Classification",
            "domain": "Natural Language Inference",
            "description": "Adversarially collected natural language inference examples",
            "example": "Premise: A man in a suit walks down the street. Hypothesis: The man is wearing formal attire. Label: Entailment",
            "num_labels": 3,
            "typical_size": {"train": 162865, "validation": 3200},
            "evaluation_metric": "Accuracy",
            "difficulty": "Very Hard",
            "why_important": "Tests robust language understanding against adversarial examples"
        },
        
        "rte": {
            "name": "Recognizing Textual Entailment",
            "task_type": "Binary Classification",
            "domain": "Natural Language Inference", 
            "description": "Determine if hypothesis follows from premise",
            "example": "Premise: The cat sat on the mat. Hypothesis: There was a cat. Label: Entailment",
            "num_labels": 2,
            "typical_size": {"train": 2490, "validation": 277},
            "evaluation_metric": "Accuracy",
            "difficulty": "Medium",
            "why_important": "Core NLU task testing logical reasoning"
        },
        
        "cb": {
            "name": "CommitmentBank",
            "task_type": "3-way Classification",
            "domain": "Natural Language Inference",
            "description": "Inference about speaker commitments in dialogue",
            "example": "Context: 'I think it might rain.' Target: 'It will rain.' Label: Neutral",
            "num_labels": 3,
            "typical_size": {"train": 250, "validation": 56},
            "evaluation_metric": "Accuracy/F1",
            "difficulty": "Hard",
            "why_important": "Tests nuanced understanding of commitments and uncertainty"
        },
        
        "copa": {
            "name": "Choice of Plausible Alternatives", 
            "task_type": "Binary Choice",
            "domain": "Causal Reasoning",
            "description": "Choose the more plausible cause or effect",
            "example": "Premise: The man broke his toe. What was the CAUSE? A) He dropped a hammer on his foot B) He put on his shoes",
            "num_labels": 2,
            "typical_size": {"train": 400, "validation": 100},
            "evaluation_metric": "Accuracy",
            "difficulty": "Medium",
            "why_important": "Tests causal reasoning and plausibility judgments"
        },
        
        # Sentiment Analysis
        "sst2": {
            "name": "Stanford Sentiment Treebank",
            "task_type": "Binary Classification",
            "domain": "Sentiment Analysis",
            "description": "Movie review sentiment classification",
            "example": "Text: 'This movie is amazing and well-crafted.' Label: Positive",
            "num_labels": 2,
            "typical_size": {"train": 67349, "validation": 872},
            "evaluation_metric": "Accuracy",
            "difficulty": "Easy-Medium",
            "why_important": "Classic sentiment analysis benchmark with fine-grained annotations"
        }
    }

def print_header():
    """Print the script header with styling."""
    print("=" * 80)
    print("🔍 DATASET EXPLORER FOR ARD-LORA TRAINING")
    print("=" * 80)
    print("This script will walk you through all datasets configured for ARD-LoRA training.")
    print("Each dataset represents a different NLP task and evaluation benchmark.")
    print("=" * 80)
    print()

def print_dataset_summary(datasets_info: Dict[str, Any]):
    """Print a summary table of all datasets."""
    print("📊 DATASET OVERVIEW")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Task Type':<20} {'Domain':<25} {'Labels':<8} {'Difficulty'}")
    print("-" * 80)
    
    for dataset_id, info in datasets_info.items():
        print(f"{dataset_id:<15} {info['task_type']:<20} {info['domain']:<25} {info['num_labels']:<8} {info['difficulty']}")
    
    print("-" * 80)
    print(f"Total datasets: {len(datasets_info)}")
    print()

def explore_dataset(dataset_id: str, info: Dict[str, Any]):
    """Provide detailed exploration of a single dataset."""
    print(f"🔬 EXPLORING: {info['name'].upper()}")
    print("=" * 60)
    
    print(f"📝 Task Type: {info['task_type']}")
    print(f"🏷️  Domain: {info['domain']}")
    print(f"🎯 Number of Labels: {info['num_labels']}")
    print(f"📏 Difficulty: {info['difficulty']}")
    print(f"📊 Evaluation Metric: {info['evaluation_metric']}")
    print()
    
    print("📖 Description:")
    print(f"   {info['description']}")
    print()
    
    print("💡 Example:")
    print(f"   {info['example']}")
    print()
    
    print("📈 Typical Dataset Size:")
    for split, size in info['typical_size'].items():
        print(f"   {split.capitalize()}: {size:,} examples")
    print()
    
    print("🎯 Why This Dataset is Important:")
    print(f"   {info['why_important']}")
    print()
    
    # Add LLaMA processing details
    print_llama_processing_details(dataset_id, info)
    
    print("🔧 ARD-LoRA Configuration:")
    print(f"   - Used for uncertainty quantification in {info['domain'].lower()}")
    print(f"   - Tests model's confidence calibration on {info['task_type'].lower()} tasks")
    print(f"   - Provides {info['difficulty'].lower()}-level challenge for ARD prior learning")
    print()

def get_task_categories(datasets_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """Group datasets by task category."""
    categories = {}
    for dataset_id, info in datasets_info.items():
        domain = info['domain']
        if domain not in categories:
            categories[domain] = []
        categories[domain].append(dataset_id)
    return categories

def print_llama_processing_details(dataset_id: str, info: Dict[str, Any]):
    """Print detailed information about LLaMA processing for this dataset."""
    print("🤖 LLAMA PROCESSING PIPELINE:")
    print("-" * 40)
    
    # Step 1: Input Formatting
    print("1️⃣  INPUT FORMATTING:")
    task_type = info['task_type']
    
    if "Multiple Choice" in task_type:
        if "4 options" in task_type:
            print("   📝 Format: Question + 4 options (A, B, C, D)")
            print("   🔤 Template: 'Question: {q}\\nA: {opt_a}\\nB: {opt_b}\\nC: {opt_c}\\nD: {opt_d}\\nAnswer:'")
            expected_tokens = "A, B, C, or D"
        else:
            print("   📝 Format: Question + 2 options (A, B)")
            print("   🔤 Template: 'Question: {q}\\nA: {opt_a}\\nB: {opt_b}\\nAnswer:'")
            expected_tokens = "A or B"
    elif "Binary Choice" in task_type:
        print("   📝 Format: Context + two alternatives")
        print("   🔤 Template: 'Context: {ctx}\\nChoice A: {choice1}\\nChoice B: {choice2}\\nAnswer:'")
        expected_tokens = "A or B"
    elif "Yes/No" in task_type:
        print("   📝 Format: Passage + Yes/No question")
        print("   🔤 Template: 'Passage: {passage}\\nQuestion: {question}\\nAnswer:'")
        expected_tokens = "Yes or No"
    elif "Classification" in task_type:
        if info['num_labels'] == 2:
            print("   📝 Format: Input text + binary label")
            print("   🔤 Template: 'Text: {text}\\nLabel:'")
            expected_tokens = "0 or 1"
        else:
            print("   📝 Format: Premise + hypothesis + entailment label")
            print("   🔤 Template: 'Premise: {premise}\\nHypothesis: {hypothesis}\\nLabel:'")
            expected_tokens = "0, 1, or 2"
    else:
        print("   📝 Format: Task-specific input structure")
        expected_tokens = "task-specific tokens"
    
    print(f"   🎯 Expected Output: {expected_tokens}")
    print()
    
    # Step 2: Tokenization
    print("2️⃣  TOKENIZATION:")
    print("   🔤 LLaMA-2-7B Tokenizer (SentencePiece, vocab_size=32,000)")
    print("   📏 Max Sequence Length: 512 tokens (configurable)")
    print("   🔧 Special Tokens:")
    print("      • <s>: Beginning of sequence (BOS)")
    print("      • </s>: End of sequence (EOS)")  
    print("      • <unk>: Unknown token")
    print("   ✂️  Truncation: From left if input > max_length")
    print("   📦 Padding: Right padding to batch max_length")
    print("   🎭 Attention Mask: 1 for real tokens, 0 for padding")
    print()
    
    # Step 3: Model Forward Pass
    print("3️⃣  LLAMA FORWARD PASS:")
    print("   🏗️  Architecture: Transformer decoder with RMSNorm + SwiGLU")
    print("   📊 Model Size: 7B parameters (6.74B base + LoRA adapters)")
    print("   🔗 Layers: 32 transformer blocks")
    print("   🧠 Attention: Multi-head attention (32 heads, 128 dim per head)")
    print("   📐 Hidden Size: 4096")
    print("   🔄 Processing Flow:")
    print("      1. Token embeddings (vocab_size × hidden_size)")
    print("      2. Positional encoding (RoPE - Rotary Position Embedding)")
    print("      3. 32 × Transformer blocks (attention + FFN)")
    print("      4. Final RMSNorm")
    print("      5. Language modeling head (hidden_size × vocab_size)")
    print()
    
    # Step 4: ARD-LoRA Integration
    print("4️⃣  ARD-LORA INTEGRATION:")
    print("   🎯 Adapter Injection: Low-rank adapters in attention layers")
    print("   📈 Rank: 16 (A: 4096×16, B: 16×4096 per attention projection)")
    print("   🎲 Bayesian Treatment: Each LoRA weight ~ N(μ, σ²)")
    print("   🔍 ARD Priors: Automatic relevance determination on LoRA weights")
    print("   📊 Forward Pass Modifications:")
    print("      • Standard: h = Wx")
    print("      • ARD-LoRA: h = Wx + (A @ B)x, where A,B ~ ARD priors")
    print("      • Multiple samples during training for uncertainty estimation")
    print()
    
    # Step 5: Output Processing
    print("5️⃣  OUTPUT PROCESSING:")
    print("   📊 Full Model Output: [batch, seq_len, vocab_size] - but we only need specific positions!")
    print("   🎯 Key Insight: We only care about logits at the answer position(s)")
    print()
    
    if dataset_id == "winogrande_s" or dataset_id == "winogrande_m":
        print("   🔍 WinoGrande Specific Processing:")
        print("      • Input: 'The trophy doesn't fit in the brown suitcase because _ is too large.'")
        print("      • Two options: 'the trophy' vs 'the suitcase'")
        print("      • We need probabilities for the BLANK POSITION, not answer tokens!")
        print("      • Extract logits at the blank position: logits[batch, blank_pos, :]")
        print("      • Get vocab indices for 'the', 'trophy' vs 'the', 'suitcase'")
        print("      • Compute: P(option1) vs P(option2) at the blank position")
        print("      • Shape: [batch, vocab_size] → [batch, 2] after option scoring")
    elif expected_tokens in ["A, B, C, or D", "A or B"]:
        print("   🔍 Multiple Choice Processing:")
        print("      • Extract logits at answer position: logits[batch, answer_pos, :]")
        print("      • Get vocab indices for tokens: 'A', 'B', 'C', 'D' (as applicable)")
        print("      • Extract relevant logits: [batch, vocab_size] → [batch, num_choices]")
        print("      • Apply softmax: P(A), P(B), P(C), P(D)")
        print("      • Prediction: argmax(choice_probabilities)")
    elif expected_tokens == "Yes or No":
        print("   🔍 Yes/No Processing:")
        print("      • Extract logits at answer position: logits[batch, answer_pos, :]")
        print("      • Get vocab indices for: 'Yes'=token_id_yes, 'No'=token_id_no")
        print("      • Extract: [logits[:, token_id_yes], logits[:, token_id_no]]")
        print("      • Shape: [batch, vocab_size] → [batch, 2]")
        print("      • Apply softmax: P(Yes), P(No)")
    elif expected_tokens in ["0 or 1", "0, 1, or 2"]:
        print("   🔍 Classification Processing:")
        print("      • Extract logits at answer position: logits[batch, answer_pos, :]")
        print(f"      • Get vocab indices for class tokens: {expected_tokens}")
        print("      • Extract relevant logits: [batch, vocab_size] → [batch, num_classes]")
        print("      • Apply softmax over class tokens")
    
    print()
    print("   ⚡ Efficiency Note:")
    print("      • Full sequence logits: [batch, seq_len, vocab_size] ≈ [32, 512, 32000] = 524M values")
    print("      • We only need: [batch, position, choice_tokens] ≈ [32, 1, 4] = 128 values")
    print("      • 99.99% of logits are irrelevant for the final prediction!")
    print()
    
    # Step 6: Loss Computation
    print("6️⃣  LOSS COMPUTATION:")
    print("   🎯 Loss Function: Cross-Entropy + ARD Regularization")
    print()
    print("   📊 Standard Cross-Entropy:")
    print("      • CE_loss = -log(P(y_true | x))")
    print("      • Where P(y_true | x) is softmax probability of correct answer")
    print(f"      • For {dataset_id}: Cross-entropy over {info['num_labels']} classes")
    print()
    print("   🎲 ARD Regularization (KL Divergence):")
    print("      • KL_loss = KL(q(θ|D) || p(θ))")
    print("      • q(θ|D): Posterior distribution of LoRA weights")
    print("      • p(θ): ARD prior (encourages sparsity)")
    print("      • KL_loss = Σ [log(σ_prior/σ_post) + (σ_post² + μ_post²)/(2σ_prior²) - 1/2]")
    print()
    print("   🔧 Total Loss:")
    print("      • Total_loss = CE_loss + β × KL_loss")
    print("      • β (KL_beta): Regularization strength (typically 0.01)")
    print("      • Balances task performance vs. parameter relevance learning")
    print()
    
    # Step 7: Gradient and Updates
    print("7️⃣  GRADIENT COMPUTATION & UPDATES:")
    print("   🔄 Backpropagation:")
    print("      • ∂Loss/∂μ (mean of LoRA weights)")
    print("      • ∂Loss/∂σ (variance of LoRA weights)")
    print("      • Only LoRA parameters updated (base LLaMA frozen)")
    print()
    print("   📈 ARD Mechanism:")
    print("      • High σ (uncertainty) → Low relevance → Pruned parameter")
    print("      • Low σ (certainty) → High relevance → Important parameter")
    print("      • Automatic pruning based on learned relevance")
    print()
    print("   ⚡ Optimizer: AdamW with:")
    print("      • Learning rate: 1e-4 (typical)")
    print("      • Weight decay: 0.01")
    print("      • Gradient clipping: 1.0")
    print()
    
    # Step 8: Uncertainty Quantification
    print("8️⃣  UNCERTAINTY QUANTIFICATION:")
    print("   🎲 Multiple Forward Passes:")
    print("      • Sample LoRA weights from learned distributions")
    print("      • N forward passes (e.g., N=25) with different weight samples")
    print("      • Collect predictions: [pred_1, pred_2, ..., pred_N]")
    print()
    print("   📊 Uncertainty Metrics:")
    print("      • Epistemic Uncertainty: Variance across samples")
    print("      • Predictive Entropy: -Σ p(y) log p(y)")
    print("      • Confidence: max(softmax_probabilities)")
    print("      • Expected Calibration Error (ECE)")
    print()
    print("   🎯 Benefits for this dataset:")
    if info['difficulty'] in ['Hard', 'Very Hard']:
        print("      • High uncertainty on difficult examples (expected)")
        print("      • Better calibration on edge cases")
        print("      • Reliable confidence estimates for wrong predictions")
    else:
        print("      • Well-calibrated confidence on straightforward examples")
        print("      • Clear separation between easy and hard instances")
        print("      • Reliable uncertainty for out-of-distribution detection")
    print()

def print_task_categories(datasets_info: Dict[str, Any]):
    """Print datasets organized by task category."""
    categories = get_task_categories(datasets_info)
    
    print("🗂️  DATASETS BY TASK CATEGORY")
    print("=" * 60)
    
    for domain, dataset_ids in categories.items():
        print(f"\n🏷️  {domain.upper()}")
        print("-" * 40)
        for dataset_id in dataset_ids:
            info = datasets_info[dataset_id]
            print(f"   • {dataset_id}: {info['name']}")
            print(f"     └─ {info['task_type']} ({info['difficulty']} difficulty)")
    print()

def print_ard_lora_context():
    """Explain how these datasets relate to ARD-LoRA training."""
    print("🎯 ARD-LORA TRAINING CONTEXT")
    print("=" * 60)
    print("ARD-LoRA (Automatic Relevance Determination LoRA) is a Bayesian approach")
    print("to parameter-efficient fine-tuning that learns which parameters are most")
    print("relevant for each task.")
    print()
    print("🔍 Why These Datasets Matter for ARD-LoRA:")
    print("   • Diverse task types test ARD's ability to identify task-specific parameters")
    print("   • Different difficulty levels challenge the uncertainty estimation")
    print("   • Varied dataset sizes test ARD prior estimation robustness") 
    print("   • Multiple domains ensure generalization of learned relevance patterns")
    print()
    print("📊 Key ARD-LoRA Benefits:")
    print("   • Automatic parameter pruning based on learned relevance")
    print("   • Uncertainty quantification for model predictions")
    print("   • Better calibration on out-of-distribution examples")
    print("   • Efficient fine-tuning with interpretable parameter importance")
    print()

def print_comprehensive_ard_lora_training():
    """Print comprehensive guide to ARD-LoRA training process."""
    print("🚀 COMPREHENSIVE ARD-LORA TRAINING GUIDE")
    print("=" * 80)
    print("This section explains the complete ARD-LoRA training pipeline from")
    print("data loading to uncertainty quantification and evaluation.")
    print("=" * 80)
    print()
    
    print("🔄 TRAINING PIPELINE OVERVIEW")
    print("-" * 50)
    print("1. Data Loading & Preprocessing")
    print("2. Model Initialization (LLaMA-2-7B + ARD-LoRA)")
    print("3. Bayesian Forward Pass")
    print("4. Loss Computation (Task + Regularization)")
    print("5. Gradient Computation & Parameter Updates")
    print("6. Uncertainty Quantification")
    print("7. Evaluation & Calibration Assessment")
    print()
    
    print("📦 1. DATA LOADING & PREPROCESSING")
    print("=" * 50)
    print("🔧 Bayesian-PEFT Dataset Loader:")
    print("   • Cached loading from Google Drive for persistence")
    print("   • Automatic subset handling (winogrande_s → small, winogrande_m → medium)")
    print("   • Task-specific formatting templates")
    print("   • Tokenization with LLaMA-2 tokenizer")
    print()
    print("📏 Preprocessing Steps:")
    print("   1. Load raw dataset from HuggingFace")
    print("   2. Apply task-specific formatting template")
    print("   3. Tokenize with max_length=512")
    print("   4. Create attention masks")
    print("   5. Extract target positions for answer tokens")
    print("   6. Batch and pad sequences")
    print()
    print("🎯 Target Extraction Example (Multiple Choice):")
    print("   Input:  'Question: What is 2+2? A: 3 B: 4 C: 5 D: 6 Answer:'")
    print("   Target: Token IDs for 'A', 'B', 'C', 'D' at answer position")
    print("   Label:  1 (corresponding to correct answer 'B')")
    print()
    
    print("🧠 2. MODEL INITIALIZATION")
    print("=" * 50)
    print("🏗️  LLaMA-2-7B Base Model:")
    print("   • 32 transformer layers")
    print("   • 4096 hidden dimensions")
    print("   • 32 attention heads")
    print("   • RMSNorm normalization")
    print("   • SwiGLU activation")
    print("   • RoPE positional encoding")
    print()
    print("🎯 ARD-LoRA Adapter Injection:")
    print("   • Rank r=16 low-rank adapters")
    print("   • Injected into: q_proj, k_proj, v_proj, o_proj")
    print("   • Each adapter: A ∈ ℝ^(d×r), B ∈ ℝ^(r×d)")
    print("   • Forward: h = W₀x + (B @ A)x")
    print()
    print("🎲 Bayesian Parameterization:")
    print("   • A ~ N(μ_A, diag(σ²_A))  # Low-rank factor A")
    print("   • B ~ N(μ_B, diag(σ²_B))  # Low-rank factor B")
    print("   • ARD priors: σ²_i ~ Gamma⁻¹(α, β) for automatic relevance")
    print()
    
    print("⚡ 3. BAYESIAN FORWARD PASS")
    print("=" * 50)
    print("🔄 Sampling Strategy:")
    print("   • Training: Single sample per forward pass")
    print("   • Evaluation: Multiple samples (N=25) for uncertainty")
    print()
    print("📊 Forward Pass Steps:")
    print("   1. Sample LoRA weights: A_sample ~ N(μ_A, σ²_A)")
    print("   2. Compute adapter output: Δh = B_sample @ A_sample @ x")
    print("   3. Add to base model: h_total = h_base + Δh")
    print("   4. Continue through remaining layers")
    print("   5. Generate logits: logits = LM_head(h_final)")
    print()
    print("🎯 Answer Token Extraction:")
    print("   • Extract logits at answer position")
    print("   • Focus on relevant vocabulary tokens (A, B, C, D, etc.)")
    print("   • Apply softmax to get probabilities")
    print()
    
    print("💰 4. LOSS COMPUTATION")
    print("=" * 50)
    print("🎯 Task Loss (Cross-Entropy):")
    print("   CE_loss = -Σᵢ yᵢ log(σ(logits_i))")
    print("   • yᵢ: One-hot encoded true label")
    print("   • σ(logits_i): Softmax probabilities")
    print("   • Measures prediction accuracy")
    print()
    print("🎲 ARD Regularization (KL Divergence):")
    print("   KL_loss = KL(q(θ|D) || p(θ))")
    print("   = Σⱼ [log(σ_prior_j/σ_post_j) + (σ²_post_j + μ²_post_j)/(2σ²_prior_j) - 1/2]")
    print("   • Encourages sparsity in LoRA parameters")
    print("   • Automatic relevance determination")
    print()
    print("⚖️  Total Loss:")
    print("   Total = CE_loss + β × KL_loss")
    print("   • β (KL_beta): Typically 0.01")
    print("   • Balances task performance vs. parameter efficiency")
    print()
    
    print("🔄 5. GRADIENT COMPUTATION & UPDATES")
    print("=" * 50)
    print("📈 Backpropagation:")
    print("   • ∂Loss/∂μ_A, ∂Loss/∂μ_B (mean parameters)")
    print("   • ∂Loss/∂σ_A, ∂Loss/∂σ_B (variance parameters)")
    print("   • Base LLaMA parameters remain frozen")
    print()
    print("🎯 ARD Mechanism:")
    print("   • High σ → Low precision → Parameter gets pruned")
    print("   • Low σ → High precision → Parameter is important")
    print("   • Automatic sparsity without manual pruning")
    print()
    print("⚡ Optimization (AdamW):")
    print("   • Learning rate: 1e-4")
    print("   • Weight decay: 0.01")
    print("   • Gradient clipping: 1.0")
    print("   • Warmup: 100 steps")
    print()
    
    print("🎲 6. UNCERTAINTY QUANTIFICATION")
    print("=" * 50)
    print("🔄 Monte Carlo Sampling:")
    print("   • Multiple forward passes with different weight samples")
    print("   • Collect predictions: P₁, P₂, ..., Pₙ")
    print("   • Aggregate for uncertainty estimates")
    print()
    print("📊 Uncertainty Metrics:")
    print("   1. Epistemic Uncertainty:")
    print("      U_epi = Var[P₁, P₂, ..., Pₙ]")
    print("      (Model's uncertainty about parameters)")
    print()
    print("   2. Predictive Entropy:")
    print("      H = -Σᵢ p̄ᵢ log(p̄ᵢ)")
    print("      where p̄ᵢ = (1/N) Σₙ Pₙᵢ")
    print()
    print("   3. Confidence Score:")
    print("      Conf = max(p̄)")
    print("      (Highest average probability)")
    print()
    print("🎯 Interpretation:")
    print("   • High uncertainty → Model is unsure → Need more data/different approach")
    print("   • Low uncertainty + correct → Model is confident and right")
    print("   • Low uncertainty + wrong → Model is confident but wrong (concerning)")
    print()
    
    print("📊 7. EVALUATION & CALIBRATION")
    print("=" * 50)
    print("🎯 Standard Metrics:")
    print("   • Accuracy: Correct predictions / Total predictions")
    print("   • F1-Score: Harmonic mean of precision and recall")
    print("   • Per-class performance for multi-class tasks")
    print()
    print("📈 Calibration Metrics:")
    print("   1. Expected Calibration Error (ECE):")
    print("      ECE = Σₘ (|Bₘ|/n) |acc(Bₘ) - conf(Bₘ)|")
    print("      Measures alignment between confidence and accuracy")
    print()
    print("   2. Reliability Diagram:")
    print("      Plot confidence vs. accuracy in bins")
    print("      Perfect calibration → diagonal line")
    print()
    print("   3. Brier Score:")
    print("      BS = (1/N) Σᵢ (pᵢ - yᵢ)²")
    print("      Lower is better (combines accuracy + calibration)")
    print()
    print("🔍 ARD-Specific Analysis:")
    print("   • Parameter relevance visualization")
    print("   • Sparsity levels achieved")
    print("   • Uncertainty-accuracy correlation")
    print("   • Task-specific parameter importance")
    print()
    
    print("💡 8. PRACTICAL TRAINING TIPS")
    print("=" * 50)
    print("🚀 Recommended Training Schedule:")
    print("   1. Warm-up phase (100 steps): Low learning rate")
    print("   2. Main training: Full learning rate")
    print("   3. Cool-down: Reduce learning rate for stability")
    print()
    print("⚙️  Hyperparameter Guidelines:")
    print("   • Start with rank=16, adjust based on performance/efficiency")
    print("   • KL_beta=0.01 as baseline, increase for more sparsity")
    print("   • Monitor KL_loss: should decrease over training")
    print("   • Uncertainty samples: 25 for evaluation, 1 for training")
    print()
    print("🔧 Troubleshooting:")
    print("   • High KL_loss: Reduce KL_beta or increase warmup")
    print("   • Poor calibration: Adjust temperature scaling post-training")
    print("   • Low sparsity: Increase KL_beta or check ARD prior settings")
    print("   • Unstable training: Reduce learning rate or add gradient clipping")
    print()
    
    print("📋 DATASET-SPECIFIC CONSIDERATIONS")
    print("=" * 50)
    print("🧠 Commonsense Reasoning (PIQA, HellaSwag, WinoGrande):")
    print("   • Expect higher uncertainty on ambiguous examples")
    print("   • ARD should identify reasoning-specific parameters")
    print("   • Monitor performance on edge cases")
    print()
    print("🔬 Science QA (ARC, OpenBookQA):")
    print("   • Look for fact-retrieval vs. reasoning parameter separation")
    print("   • Higher uncertainty on multi-step problems")
    print("   • Different parameter relevance for easy vs. hard subsets")
    print()
    print("🔗 NLI Tasks (ANLI, RTE, CB):")
    print("   • Expect task-specific parameter patterns")
    print("   • High uncertainty on adversarial examples (ANLI)")
    print("   • Calibration especially important for entailment decisions")
    print()
    print("💭 Sentiment Analysis (SST2):")
    print("   • Should achieve good calibration (easier task)")
    print("   • Use as baseline for comparing ARD effectiveness")
    print("   • Monitor uncertainty on neutral/ambiguous sentiment")
    print()
    
    print("=" * 80)
    print("🎓 This comprehensive guide covers the complete ARD-LoRA training")
    print("pipeline from data preprocessing to uncertainty quantification.")
    print("Use this as a reference for understanding and debugging your training.")
    print("=" * 80)

def interactive_exploration(datasets_info: Dict[str, Any]):
    """Interactive dataset exploration."""
    print("🔍 INTERACTIVE EXPLORATION")
    print("=" * 60)
    print("Choose an option:")
    print()
    
    # Show numbered list of datasets
    dataset_list = list(datasets_info.keys())
    for i, dataset_id in enumerate(dataset_list, 1):
        info = datasets_info[dataset_id]
        print(f"{i:2d}. {dataset_id:<15} - {info['name']}")
    
    print()
    print("Special Options:")
    print(f"{len(dataset_list)+1:2d}. 🚀 Complete ARD-LoRA Training Guide")
    print(f"{len(dataset_list)+2:2d}. 🎯 ARD-LoRA Quick Reference")
    print("  q. Quit")
    print()
    
    while True:
        try:
            choice = input(f"Enter number (1-{len(dataset_list)+2}) or 'q' to quit: ").strip().lower()
            
            if choice in ['quit', 'q', 'exit']:
                break
            
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(dataset_list):
                dataset_id = dataset_list[choice_num - 1]
                print("\n" + "="*80)
                explore_dataset(dataset_id, datasets_info[dataset_id])
                print("="*80 + "\n")
                
            elif choice_num == len(dataset_list) + 1:
                print("\n" + "="*80)
                print_comprehensive_ard_lora_training()
                print("="*80 + "\n")
                
            elif choice_num == len(dataset_list) + 2:
                print("\n" + "="*80)
                print_ard_lora_quick_reference()
                print("="*80 + "\n")
                
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(dataset_list)+2}.")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def print_training_recommendations():
    """Print recommendations for ARD-LoRA training with these datasets."""
    print("💡 TRAINING RECOMMENDATIONS")
    print("=" * 60)
    print("Based on the dataset characteristics, here are recommendations for")
    print("ARD-LoRA training:")
    print()
    print("🚀 Recommended Training Order:")
    print("   1. Start with SST2 (sentiment) - easier task, good for debugging")
    print("   2. Try PIQA or BoolQ - medium difficulty, good dataset sizes") 
    print("   3. Progress to ARC-Easy - science reasoning with clear structure")
    print("   4. Challenge with HellaSwag or ANLI - harder reasoning tasks")
    print("   5. Test on WinoGrande variants - pronoun resolution edge cases")
    print()
    print("⚙️  ARD-LoRA Hyperparameter Suggestions:")
    print("   • Rank: 16 (good balance of capacity and efficiency)")
    print("   • KL Beta: 0.01 (moderate regularization for ARD prior learning)")
    print("   • ARD Prior Samples: 25 (sufficient for variance estimation)")
    print("   • Max Length: 512 (handles most examples while saving memory)")
    print()
    print("📈 Expected Performance Patterns:")
    print("   • Easy tasks (SST2, ARC-Easy): High accuracy, good calibration")
    print("   • Medium tasks (PIQA, BoolQ): Moderate accuracy, decent uncertainty")
    print("   • Hard tasks (HellaSwag, ANLI): Lower accuracy, higher uncertainty")
    print("   • Small datasets (CB, COPA): May need more regularization")
    print()

def print_ard_lora_quick_reference():
    """Print a quick reference guide for ARD-LoRA key concepts."""
    print("🎯 ARD-LORA QUICK REFERENCE")
    print("=" * 60)
    print("Essential concepts and formulas for ARD-LoRA training.")
    print()
    
    print("🧠 CORE CONCEPTS:")
    print("-" * 30)
    print("• ARD: Automatic Relevance Determination")
    print("• LoRA: Low-Rank Adaptation")
    print("• Bayesian: Each parameter has mean (μ) and variance (σ²)")
    print("• Sparsity: High variance → Low relevance → Pruned parameter")
    print()
    
    print("📊 KEY FORMULAS:")
    print("-" * 30)
    print("Forward Pass:")
    print("   h = W₀x + (B @ A)x")
    print("   where A ~ N(μₐ, σ²ₐ), B ~ N(μᵦ, σ²ᵦ)")
    print()
    print("Total Loss:")
    print("   L = CrossEntropy + β × KL_Divergence")
    print("   L = -log P(y|x) + β × KL(q(θ) || p(θ))")
    print()
    print("KL Divergence (per parameter):")
    print("   KL = log(σ_prior/σ_post) + (σ²_post + μ²_post)/(2σ²_prior) - 1/2")
    print()
    print("Uncertainty (Epistemic):")
    print("   U = Var[P₁, P₂, ..., Pₙ] across N samples")
    print()
    
    print("⚙️  DEFAULT HYPERPARAMETERS:")
    print("-" * 30)
    print("• Rank (r): 16")
    print("• KL Beta (β): 0.01")
    print("• Learning Rate: 1e-4")
    print("• Max Length: 512 tokens")
    print("• Uncertainty Samples: 25")
    print("• Optimizer: AdamW")
    print("• Weight Decay: 0.01")
    print()
    
    print("🎯 TASK-SPECIFIC OUTPUTS:")
    print("-" * 30)
    print("Multiple Choice (4-way): Extract logits for A, B, C, D tokens")
    print("Multiple Choice (2-way): Extract logits for A, B tokens")
    print("Yes/No: Extract logits for 'Yes', 'No' tokens")
    print("Classification: Extract logits for class tokens (0, 1, 2, ...)")
    print("Binary Sentiment: Extract logits for positive/negative tokens")
    print()
    
    print("📈 EVALUATION METRICS:")
    print("-" * 30)
    print("• Accuracy: Standard task performance")
    print("• ECE: Expected Calibration Error (confidence-accuracy alignment)")
    print("• Brier Score: Combines accuracy and calibration")
    print("• Epistemic Uncertainty: Model parameter uncertainty")
    print("• Sparsity: Percentage of pruned parameters")
    print()
    
    print("🔧 TROUBLESHOOTING:")
    print("-" * 30)
    print("High KL Loss → Reduce β or increase warmup")
    print("Poor Calibration → Add temperature scaling")
    print("Low Sparsity → Increase β")
    print("Unstable Training → Reduce LR or add grad clipping")
    print("High Uncertainty → Normal for hard tasks, check if calibrated")
    print()
    
    print("🚀 TRAINING WORKFLOW:")
    print("-" * 30)
    print("1. Load dataset with task-specific formatting")
    print("2. Initialize LLaMA-2-7B + ARD-LoRA adapters")
    print("3. Forward pass: Sample weights, compute logits")
    print("4. Loss: CrossEntropy + β × KL regularization")
    print("5. Backward: Update μ and σ parameters only")
    print("6. Eval: Multiple samples for uncertainty quantification")
    print("7. Analysis: Check sparsity, calibration, task performance")
    print()

def main():
    """Main function to run the dataset explorer."""
    print_header()
    
    # Get dataset information
    datasets_info = get_dataset_info()
    
    # Print overview
    print_dataset_summary(datasets_info)
    
    # Print task categories
    print_task_categories(datasets_info)
    
    # Print ARD-LoRA context
    print_ard_lora_context()
    
    # Print training recommendations
    print_training_recommendations()
    
    # Interactive exploration
    try:
        interactive_exploration(datasets_info)
    except KeyboardInterrupt:
        print("\nExiting dataset explorer...")
    
    print("\n" + "="*80)
    print("🎓 DATASET EXPLORATION COMPLETE")
    print("="*80)
    print("You now have a comprehensive understanding of all datasets configured")
    print("for ARD-LoRA training. Use this knowledge to make informed decisions")
    print("about training order, hyperparameters, and expected performance.")
    print("="*80)

if __name__ == "__main__":
    main()
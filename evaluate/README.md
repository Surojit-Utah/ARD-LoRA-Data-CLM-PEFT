# Uncertainty Evaluation for ARD-LoRA Models

This directory contains scripts and utilities for evaluating the uncertainty estimation quality of ARD-LoRA models using standard metrics.

## 📊 Evaluation Metrics

The evaluation system implements three key uncertainty metrics:

### 1. **Accuracy (ACC)**
- **Definition**: Standard classification accuracy
- **Formula**: `ACC = (Correct Predictions) / (Total Predictions)`
- **Range**: [0, 1] (higher is better)
- **Purpose**: Measures basic predictive performance

### 2. **Expected Calibration Error (ECE)** 
- **Definition**: Measures how well model confidence matches actual accuracy
- **Formula**: `ECE = Σ (|Bₘ|/N) × |acc(Bₘ) - conf(Bₘ)|`
  - `Bₘ`: Set of samples in confidence bin m
  - `|Bₘ|`: Number of samples in bin m  
  - `acc(Bₘ)`: Accuracy of samples in bin m
  - `conf(Bₘ)`: Average confidence in bin m
- **Range**: [0, 1] (lower is better)
- **Purpose**: Evaluates confidence calibration quality

### 3. **Negative Log-Likelihood (NLL)**
- **Definition**: Measures quality of predicted probability distributions
- **Formula**: `NLL = -1/N × Σ log P_θ(yₙ)`
- **Range**: [0, ∞) (lower is better)
- **Purpose**: Assesses predictive probability quality

## 🚀 Quick Start

### Basic Usage

```bash
# Evaluate a trained model
python evaluate/evaluate_model.py --model_path ./outputs/model --config alpaca

# With custom parameters
python evaluate/evaluate_model.py \
    --model_path ./outputs/model \
    --config alpaca \
    --num_samples 2000 \
    --batch_size 8 \
    --n_bins 20
```

### Programmatic Usage

```python
from evaluate import UncertaintyEvaluator, print_evaluation_results

# Initialize evaluator
evaluator = UncertaintyEvaluator(n_bins=15)

# Evaluate predictions
metrics = evaluator.evaluate_predictions(y_true, y_probs)

# Print formatted results
print_evaluation_results(metrics)
```

## 📁 Directory Structure

```
evaluate/
├── __init__.py                 # Package initialization
├── uncertainty_metrics.py     # Core evaluation metrics implementation
├── evaluate_model.py          # Integrated model evaluation script
└── README.md                  # This documentation
```

## 🔧 Core Components

### `UncertaintyEvaluator` Class

The main evaluation class that implements all uncertainty metrics:

```python
class UncertaintyEvaluator:
    def __init__(self, n_bins: int = 15):
        """Initialize with specified number of ECE bins"""
        
    def compute_accuracy(self, y_true, y_pred) -> float:
        """Compute classification accuracy"""
        
    def compute_nll(self, y_true, y_probs) -> float:
        """Compute Negative Log-Likelihood"""
        
    def compute_ece(self, y_true, y_probs) -> float:
        """Compute Expected Calibration Error"""
        
    def evaluate_predictions(self, y_true, y_probs) -> Dict[str, float]:
        """Compute all metrics at once"""
```

### `evaluate_ard_lora_model` Function

Specialized evaluation function for ARD-LoRA models:

- Handles causal language modeling evaluation
- Supports batched processing for memory efficiency
- Integrates with existing ARD-LoRA pipeline
- Properly handles token-level predictions

## 📋 Command Line Arguments

### `evaluate_model.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | **Required** | Path to trained model directory |
| `--config` | str | `alpaca` | Dataset configuration name |
| `--base_model` | str | `NousResearch/Llama-2-7b-hf` | Base LLaMA model |
| `--num_samples` | int | `1000` | Number of token predictions to evaluate |
| `--n_bins` | int | `15` | Number of bins for ECE calculation |
| `--batch_size` | int | `4` | Evaluation batch size |
| `--output` | str | `model_path/evaluation_results.json` | Output path for results |

### Supported Dataset Configurations

The evaluation supports all dataset configurations from `dataloader/bayesian_peft_cached.py`:

- `alpaca` - Alpaca instruction following
- `alpaca_gpt4` - GPT-4 generated Alpaca data
- `code_alpaca` - Code generation tasks
- `dolly` - Databricks Dolly dataset
- `oasst1` - Open Assistant conversations
- `gpteacher` - GPT-Teacher dataset

## 📈 Example Output

```
============================================================
📊 UNCERTAINTY EVALUATION RESULTS
============================================================
📈 **Accuracy (ACC)**:     0.3245
   → Higher is better (perfect = 1.0)

📉 **Negative Log-Likelihood (NLL)**: 6.7823
   → Lower is better (perfect = 0.0)
   → Measures predictive probability quality

🎯 **Expected Calibration Error (ECE)**: 0.1245
   → Lower is better (perfect = 0.0)
   → Measures confidence calibration quality

📊 **Dataset Info**:
   → Evaluated samples: 1,000
   → Vocabulary size: 32,000
============================================================
```

## 🔬 Technical Details

### Causal Language Modeling Evaluation

For causal language models like LLaMA, the evaluation:

1. **Shifts predictions**: Predicts token `i+1` given tokens `0...i`
2. **Handles masking**: Ignores padding tokens (labels = -100)
3. **Token-level metrics**: Evaluates individual token predictions
4. **Batched processing**: Memory-efficient evaluation on large vocabularies

### Memory Considerations

- Uses 4-bit quantization for efficient memory usage
- Processes evaluation in batches to handle large datasets
- Limits token predictions per batch to prevent OOM errors
- Supports CPU fallback for systems without CUDA

### Calibration Quality Assessment

ECE measures calibration by:

1. **Binning**: Groups predictions by confidence level (0-1 range)
2. **Accuracy calculation**: Computes accuracy within each bin
3. **Weighted difference**: Measures |confidence - accuracy| per bin
4. **Final ECE**: Weighted average across all bins

## 🧪 Testing and Validation

### Synthetic Data Testing

The scripts include synthetic data generation for testing:

```python
# Generate synthetic evaluation data
np.random.seed(42)
n_samples = 1000
n_classes = 1000  # Vocabulary size

y_true = np.random.randint(0, n_classes, n_samples)
logits = np.random.randn(n_samples, n_classes) * 2.0
y_probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
```

### Integration Testing

Test the complete pipeline:

```bash
# Run with synthetic data
python evaluate/uncertainty_metrics.py

# Test specific model evaluation
python evaluate/evaluate_model.py --model_path ./outputs/test_model --config alpaca
```

## 🔗 Integration with Training Pipeline

The evaluation integrates seamlessly with the ARD-LoRA training pipeline:

1. **Model Loading**: Automatically loads ARD-LoRA models with ProbLoRA layers
2. **Dataset Integration**: Uses the same cached datasets as training
3. **Configuration**: Supports all dataset configurations
4. **Output Format**: Results saved as JSON for further analysis

### Post-Training Evaluation

Add evaluation to your training scripts:

```python
from evaluate import evaluate_ard_lora_model_integrated

# After training completion
metrics = evaluate_ard_lora_model_integrated(
    model=trained_model,
    eval_dataloader=eval_dataloader,
    tokenizer=tokenizer,
    device=device
)
```

## 🎯 Best Practices

### Evaluation Setup

1. **Sample Size**: Use 1000+ token predictions for reliable metrics
2. **Bin Count**: Use 10-20 bins for ECE (15 is standard)
3. **Batch Size**: Adjust based on GPU memory (4-8 typical)
4. **Multiple Runs**: Average results across multiple evaluation runs

### Interpretation Guidelines

- **High Accuracy + Low ECE**: Well-calibrated, high-quality model
- **High Accuracy + High ECE**: Good predictions but overconfident
- **Low Accuracy + Low ECE**: Poor model but well-calibrated
- **Low NLL**: Model assigns high probability to correct predictions

### Performance Optimization

- Use GPU when available for faster evaluation
- Enable mixed precision for memory efficiency
- Process in batches to handle large datasets
- Cache datasets to avoid repeated loading

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `--batch_size` or `--num_samples`
   - Use CPU evaluation: `CUDA_VISIBLE_DEVICES="" python evaluate_model.py`

2. **Model Loading Errors**:
   - Verify model path contains checkpoints
   - Check base model compatibility
   - Ensure ProbLoRA layers are properly injected

3. **Dataset Loading Issues**:
   - Verify dataset configuration name
   - Check cached dataset availability
   - Ensure proper tokenizer setup

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH="${PYTHONPATH}:."
python -v evaluate/evaluate_model.py --model_path ./outputs/model
```

## 📚 References

1. Guo, C., et al. "On calibration of modern neural networks." ICML 2017.
2. Niculescu-Mizil, A., & Caruana, R. "Predicting good probabilities with supervised learning." ICML 2005.
3. Ovadia, Y., et al. "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift." NeurIPS 2019.
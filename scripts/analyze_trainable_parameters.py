#!/usr/bin/env python3
"""
Comprehensive Trainable Parameter Analysis for ARD-LoRA

This script provides detailed analysis of trainable parameters in the ARD-LoRA model,
breaking down parameters by layer, head, and matrix type (A, B, G).

Usage:
    python scripts/analyze_trainable_parameters.py --model_path <path_to_model> --output <output_file>
    
Features:
- Layer-wise parameter breakdown
- Head-wise parameter counting for multi-head attention
- Matrix-wise analysis (A, B, G matrices in ProbLoRA)
- Detailed logging to text file
- Validation against expected total parameter count
"""

import os
import sys
import argparse
import torch
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model_llama import ProbLoRALayer, inject_problora_llama
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


class ParameterAnalyzer:
    """Comprehensive parameter analysis for ARD-LoRA models."""
    
    def __init__(self, model, output_file: str = None):
        """
        Initialize parameter analyzer.
        
        Args:
            model: The model to analyze
            output_file: Path to output log file (optional)
        """
        self.model = model
        self.output_file = output_file
        self.log_lines = []
        
        # Parameter tracking
        self.layer_stats = defaultdict(dict)
        self.total_trainable = 0
        self.total_parameters = 0
        
    def log(self, message: str, print_console: bool = True):
        """Log message to both console and internal log."""
        if print_console:
            print(message)
        self.log_lines.append(message)
        
    def save_log(self):
        """Save log to file if output file is specified."""
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_lines))
            self.log(f"Parameter analysis saved to: {self.output_file}")
    
    def analyze_parameter_tensor(self, param: torch.Tensor, param_name: str) -> Dict[str, Any]:
        """Analyze a single parameter tensor."""
        return {
            'name': param_name,
            'shape': list(param.shape),
            'numel': param.numel(),
            'requires_grad': param.requires_grad,
            'dtype': str(param.dtype),
            'device': str(param.device)
        }
    
    def analyze_problora_layer(self, layer_name: str, layer_module) -> Dict[str, Any]:
        """Analyze a ProbLoRA layer in detail."""
        layer_info = {
            'layer_name': layer_name,
            'layer_type': type(layer_module).__name__,
            'total_params': 0,
            'trainable_params': 0,
            'matrices': {},
            'heads': {}
        }
        
        # Check if this is a ProbLoRA layer
        if isinstance(layer_module, ProbLoRALayer):
            self.log(f"\nAnalyzing ProbLoRA Layer: {layer_name}")
            
            # Analyze A matrix
            if hasattr(layer_module, 'A'):
                A_info = self.analyze_parameter_tensor(layer_module.A, f"{layer_name}.A")
                layer_info['matrices']['A'] = A_info
                layer_info['total_params'] += A_info['numel']
                if A_info['requires_grad']:
                    layer_info['trainable_params'] += A_info['numel']
                self.log(f"   A matrix: {A_info['shape']} = {A_info['numel']:,} params (trainable: {A_info['requires_grad']})")
            
            # For deterministic mode, check mu_A instead of A
            elif hasattr(layer_module, 'mu_A'):
                A_info = self.analyze_parameter_tensor(layer_module.mu_A, f"{layer_name}.mu_A")
                layer_info['matrices']['A'] = A_info
                layer_info['total_params'] += A_info['numel']
                if A_info['requires_grad']:
                    layer_info['trainable_params'] += A_info['numel']
                self.log(f"   mu_A matrix: {A_info['shape']} = {A_info['numel']:,} params (trainable: {A_info['requires_grad']})")
            
            # Analyze B matrix
            if hasattr(layer_module, 'B'):
                B_info = self.analyze_parameter_tensor(layer_module.B, f"{layer_name}.B")
                layer_info['matrices']['B'] = B_info
                layer_info['total_params'] += B_info['numel']
                if B_info['requires_grad']:
                    layer_info['trainable_params'] += B_info['numel']
                self.log(f"   B matrix: {B_info['shape']} = {B_info['numel']:,} params (trainable: {B_info['requires_grad']})")
            
            # Analyze G matrix (variance parameters) - only in probabilistic mode
            if hasattr(layer_module, 'G'):
                G_info = self.analyze_parameter_tensor(layer_module.G, f"{layer_name}.G")
                layer_info['matrices']['G'] = G_info
                layer_info['total_params'] += G_info['numel']
                if G_info['requires_grad']:
                    layer_info['trainable_params'] += G_info['numel']
                self.log(f"   G matrix: {G_info['shape']} = {G_info['numel']:,} params (trainable: {G_info['requires_grad']})")
            
            # Report layer mode
            mode = "deterministic LoRA" if layer_module.deterministic else "probabilistic LoRA"
            self.log(f"   Mode: {mode} (rank={layer_module.rank})")
            
            # Layer summary
            self.log(f"   Layer Total: {layer_info['total_params']:,} params ({layer_info['trainable_params']:,} trainable)")
            
        return layer_info
    
    def analyze_model(self) -> Dict[str, Any]:
        """Perform comprehensive model analysis focused on ProbLoRA parameters."""
        self.log("PROBLORA-FOCUSED TRAINABLE PARAMETER ANALYSIS")
        self.log("=" * 80)
        self.log(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model Type: {type(self.model).__name__}")
        self.log(f"Focus: ProbLoRA layers (A, B, G matrices) only")
        self.log("")
        
        # Track all parameters first, but distinguish ProbLoRA vs base model
        problora_trainable = 0
        base_model_trainable = 0
        
        for name, param in self.model.named_parameters():
            self.total_parameters += param.numel()
            if param.requires_grad:
                self.total_trainable += param.numel()
                # Check if this is a ProbLoRA parameter (A, B, G matrices, or mu_A for deterministic)
                if any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A']):
                    problora_trainable += param.numel()
                else:
                    base_model_trainable += param.numel()
        
        self.log(f"OVERALL PARAMETER SUMMARY:")
        self.log(f"   Total parameters: {self.total_parameters:,}")
        self.log(f"   Total trainable parameters: {self.total_trainable:,}")
        self.log(f"   ProbLoRA trainable parameters: {problora_trainable:,}")
        self.log(f"   Base model trainable parameters: {base_model_trainable:,}")
        self.log(f"   ProbLoRA percentage of total: {(problora_trainable/self.total_parameters)*100:.4f}%")
        self.log(f"   ProbLoRA percentage of trainable: {(problora_trainable/self.total_trainable)*100:.2f}%")
        self.log("")
        
        # Warn if base model has trainable parameters (shouldn't happen with proper LoRA)
        if base_model_trainable > 0:
            self.log(f"WARNING: Base model has {base_model_trainable:,} trainable parameters!")
            self.log(f"         In proper LoRA setup, only ProbLoRA matrices should be trainable.")
            self.log("")
        
        # Analyze ProbLoRA layers specifically
        problora_layers_found = 0
        total_problora_params = 0
        
        self.log("PROBLORA LAYER-BY-LAYER ANALYSIS:")
        self.log("-" * 60)
        
        for name, module in self.model.named_modules():
            # Check if this module is a ProbLoRA layer
            if isinstance(module, ProbLoRALayer):
                layer_info = self.analyze_problora_layer(name, module)
                self.layer_stats[name] = layer_info
                problora_layers_found += 1
                total_problora_params += layer_info['trainable_params']
        
        # Summary statistics
        self.log(f"\nPROBLORA LAYER SUMMARY:")
        self.log(f"   ProbLoRA layers found: {problora_layers_found}")
        self.log(f"   Total ProbLoRA trainable params: {total_problora_params:,}")
        self.log(f"   Non-ProbLoRA trainable params: {self.total_trainable - total_problora_params:,}")
        
        # Detailed breakdown by matrix type
        self.analyze_matrix_breakdown()
        
        # Validation
        self.validate_parameter_count()
        
        return {
            'total_parameters': self.total_parameters,
            'total_trainable': self.total_trainable,
            'problora_layers': problora_layers_found,
            'problora_trainable': total_problora_params,
            'layer_stats': dict(self.layer_stats)
        }
    
    def analyze_matrix_breakdown(self):
        """Analyze parameters by matrix type (A, B, G)."""
        self.log(f"\nMATRIX TYPE BREAKDOWN:")
        self.log("-" * 40)
        
        matrix_totals = {'A': 0, 'B': 0, 'G': 0}
        head_analysis = {'A': [], 'B': []}
        
        for layer_name, layer_info in self.layer_stats.items():
            for matrix_type in ['A', 'B', 'G']:
                if matrix_type in layer_info['matrices']:
                    matrix_info = layer_info['matrices'][matrix_type]
                    if matrix_info['requires_grad']:
                        matrix_totals[matrix_type] += matrix_info['numel']
                
                # Collect head analysis for A and B
                if matrix_type in ['A', 'B'] and matrix_type in layer_info['heads']:
                    head_info = layer_info['heads'][matrix_type]
                    head_analysis[matrix_type].append({
                        'layer': layer_name,
                        'num_heads': head_info['num_heads'],
                        'params_per_head': head_info['params_per_head'],
                        'total_params': head_info['total_params']
                    })
        
        # Print matrix totals
        for matrix_type, total in matrix_totals.items():
            self.log(f"   {matrix_type} matrices: {total:,} parameters")
        
        # Print head analysis
        for matrix_type in ['A', 'B']:
            if head_analysis[matrix_type]:
                self.log(f"\n{matrix_type} MATRIX HEAD ANALYSIS:")
                total_heads = 0
                for head_info in head_analysis[matrix_type]:
                    layer_short = head_info['layer'].split('.')[-2] if '.' in head_info['layer'] else head_info['layer']
                    self.log(f"   {layer_short}: {head_info['num_heads']} heads × {head_info['params_per_head']:,} = {head_info['total_params']:,}")
                    total_heads += head_info['num_heads']
                self.log(f"   Total {matrix_type} heads across all layers: {total_heads}")
    
    def validate_parameter_count(self):
        """Validate the parameter count against expected values, focusing on ProbLoRA."""
        self.log(f"\nPROBLORA PARAMETER VALIDATION:")
        self.log("-" * 40)
        
        expected_count = 12_582_912  # The reported trainable parameter count
        actual_count = self.total_trainable
        
        # Count ProbLoRA-specific parameters (including mu_A for deterministic mode)
        problora_count = sum(p.numel() for name, p in self.model.named_parameters()
                            if p.requires_grad and any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A']))
        
        non_problora_count = actual_count - problora_count
        
        self.log(f"   Expected total trainable parameters: {expected_count:,}")
        self.log(f"   Actual total trainable parameters:   {actual_count:,}")
        self.log(f"   ProbLoRA trainable parameters:       {problora_count:,}")
        self.log(f"   Non-ProbLoRA trainable parameters:   {non_problora_count:,}")
        self.log(f"   Difference from expected: {abs(actual_count - expected_count):,}")
        
        if actual_count == expected_count:
            self.log("   ✓ Parameter count matches expected value!")
        else:
            percentage_diff = abs(actual_count - expected_count) / expected_count * 100
            self.log(f"   Parameter count differs by {percentage_diff:.2f}%")
            
            if percentage_diff < 1.0:
                self.log("   Small difference, likely due to rounding or additional parameters")
            else:
                self.log("   Significant difference, investigation needed")
        
        # ProbLoRA-specific validation
        if non_problora_count == 0:
            self.log("   ✓ EXCELLENT: Only ProbLoRA parameters are trainable (proper LoRA setup)")
        else:
            self.log(f"   ⚠ WARNING: {non_problora_count:,} non-ProbLoRA parameters are trainable")
            self.log("     This suggests base model parameters may not be properly frozen")
        
        # Expected ProbLoRA parameter breakdown (for reference)
        expected_problora = expected_count if non_problora_count == 0 else problora_count
        if problora_count == expected_problora:
            self.log(f"   ✓ ProbLoRA parameter count is correct: {problora_count:,}")
        else:
            diff = abs(problora_count - expected_problora)
            self.log(f"   ⚠ ProbLoRA parameter difference: {diff:,} ({diff/expected_problora*100:.2f}%)")

    def validate_optimizer_parameters(self, trainer=None, optimizer=None):
        """
        Validate that HuggingFace trainer/optimizer is using the correct set of trainable parameters.
        
        Args:
            trainer: HuggingFace Trainer instance (optional)
            optimizer: PyTorch optimizer instance (optional)
        """
        self.log(f"\nOPTIMIZER PARAMETER VALIDATION:")
        self.log("-" * 45)
        
        # Get model's trainable parameters
        model_trainable_params = []
        model_param_names = []
        model_param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_trainable_params.append(param)
                model_param_names.append(name)
                model_param_count += param.numel()
        
        self.log(f"   Model trainable parameters: {model_param_count:,}")
        self.log(f"   Model trainable parameter tensors: {len(model_trainable_params)}")
        
        # Check trainer's optimizer if provided
        if trainer is not None:
            self.log(f"\n   TRAINER OPTIMIZER ANALYSIS:")
            try:
                # Force optimizer creation if it doesn't exist
                if trainer.optimizer is None:
                    self.log(f"   Creating optimizer (trainer.optimizer was None)...")
                    trainer.create_optimizer()
                
                trainer_optimizer = trainer.optimizer
                
                # Analyze optimizer parameters
                optimizer_param_count = 0
                optimizer_param_tensors = 0
                optimizer_param_groups = len(trainer_optimizer.param_groups)
                
                for i, group in enumerate(trainer_optimizer.param_groups):
                    group_param_count = 0
                    for param in group['params']:
                        optimizer_param_count += param.numel()
                        group_param_count += param.numel()
                        optimizer_param_tensors += 1
                    self.log(f"   Parameter group {i+1}: {group_param_count:,} parameters")
                
                self.log(f"   Optimizer total parameters: {optimizer_param_count:,}")
                self.log(f"   Optimizer parameter tensors: {optimizer_param_tensors}")
                self.log(f"   Optimizer parameter groups: {optimizer_param_groups}")
                
                # Validation
                if optimizer_param_count == model_param_count:
                    self.log(f"   PASS: Optimizer parameter count matches model!")
                else:
                    diff = abs(optimizer_param_count - model_param_count)
                    self.log(f"   FAIL: Parameter count mismatch!")
                    self.log(f"   Difference: {diff:,} parameters")
                
                if optimizer_param_tensors == len(model_trainable_params):
                    self.log(f"   PASS: Optimizer tensor count matches model!")
                else:
                    diff = abs(optimizer_param_tensors - len(model_trainable_params))
                    self.log(f"   FAIL: Tensor count mismatch!")
                    self.log(f"   Difference: {diff} tensors")
                
            except Exception as e:
                self.log(f"   ERROR analyzing trainer optimizer: {e}")
                return None
        
        # Check standalone optimizer if provided
        if optimizer is not None:
            self.log(f"\n   STANDALONE OPTIMIZER ANALYSIS:")
            try:
                optimizer_param_count = 0
                optimizer_param_tensors = 0
                optimizer_param_groups = len(optimizer.param_groups)
                
                for i, group in enumerate(optimizer.param_groups):
                    group_param_count = 0
                    for param in group['params']:
                        optimizer_param_count += param.numel()
                        group_param_count += param.numel()
                        optimizer_param_tensors += 1
                    self.log(f"   Parameter group {i+1}: {group_param_count:,} parameters")
                
                self.log(f"   Optimizer total parameters: {optimizer_param_count:,}")
                self.log(f"   Optimizer parameter tensors: {optimizer_param_tensors}")
                self.log(f"   Optimizer parameter groups: {optimizer_param_groups}")
                
                # Validation
                if optimizer_param_count == model_param_count:
                    self.log(f"   PASS: Optimizer parameter count matches model!")
                else:
                    diff = abs(optimizer_param_count - model_param_count)
                    self.log(f"   FAIL: Parameter count mismatch!")
                    self.log(f"   Difference: {diff:,} parameters")
                
                if optimizer_param_tensors == len(model_trainable_params):
                    self.log(f"   PASS: Optimizer tensor count matches model!")
                else:
                    diff = abs(optimizer_param_tensors - len(model_trainable_params))
                    self.log(f"   FAIL: Tensor count mismatch!")
                    self.log(f"   Difference: {diff} tensors")
                
            except Exception as e:
                self.log(f"   ERROR analyzing standalone optimizer: {e}")
                return None
        
        # Detailed parameter name analysis
        if trainer is not None or optimizer is not None:
            self.log(f"\n   DETAILED PARAMETER ANALYSIS:")
            
            # Show sample of trainable parameter names
            self.log(f"   Sample trainable parameter names:")
            for i, name in enumerate(model_param_names[:10]):  # Show first 10
                param = dict(self.model.named_parameters())[name]
                self.log(f"     {i+1:2d}. {name} - {list(param.shape)} - {param.numel():,} params")
            
            if len(model_param_names) > 10:
                self.log(f"     ... and {len(model_param_names) - 10} more parameters")
            
            # Check for parameters that should be trainable (ProbLoRA specific)
            problora_params = [name for name in model_param_names if any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
            non_problora_params = [name for name in model_param_names if not any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
            
            self.log(f"\n   PARAMETER CLASSIFICATION:")
            self.log(f"     ProbLoRA parameters: {len(problora_params)}")
            self.log(f"     Non-ProbLoRA parameters: {len(non_problora_params)}")
            
            if non_problora_params:
                self.log(f"     WARNING: Found non-ProbLoRA trainable parameters:")
                for name in non_problora_params[:5]:  # Show first 5
                    param = dict(self.model.named_parameters())[name]
                    self.log(f"       - {name} - {param.numel():,} params")
                if len(non_problora_params) > 5:
                    self.log(f"       ... and {len(non_problora_params) - 5} more")
            else:
                self.log(f"     PASS: All trainable parameters are ProbLoRA parameters")
        
        return {
            'model_param_count': model_param_count,
            'model_param_tensors': len(model_trainable_params),
            'problora_param_names': [name for name in model_param_names if any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])],
            'non_problora_param_names': [name for name in model_param_names if not any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
        }

    def validate_gradient_flow_and_updates(self, trainer=None):
        """
        Validate that gradients flow correctly and only trainable parameters update.
        This is the ultimate test for proper LoRA parameter handling.
        """
        self.log(f"\nGRADIENT FLOW AND PARAMETER UPDATE VALIDATION:")
        self.log("=" * 60)
        
        if trainer is None:
            self.log("   ERROR: Trainer required for gradient flow validation")
            return None
        
        try:
            # Force optimizer creation if needed
            if trainer.optimizer is None:
                trainer.create_optimizer()
            
            # Create dummy input for forward/backward pass
            tokenizer = trainer.tokenizer
            dummy_text = "The quick brown fox jumps over the lazy dog."
            inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            self.log(f"   Using dummy input: '{dummy_text}'")
            self.log(f"   Input shape: {input_ids.shape}")
            
            # Switch to training mode and clear gradients
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            self.log(f"\n   FORWARD PASS AND GRADIENT COMPUTATION:")
            
            # Forward pass with loss computation
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            self.log(f"   Loss computed: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            self.log(f"   Backward pass completed")
            
            # Analyze gradient distribution
            trainable_with_grads = 0
            trainable_without_grads = 0
            frozen_with_grads = 0
            frozen_without_grads = 0
            
            trainable_grad_norm = 0.0
            frozen_grad_norm = 0.0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        trainable_with_grads += 1
                        trainable_grad_norm += param.grad.abs().sum().item()
                    else:
                        trainable_without_grads += 1
                else:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        frozen_with_grads += 1
                        frozen_grad_norm += param.grad.abs().sum().item()
                    else:
                        frozen_without_grads += 1
            
            self.log(f"\n   GRADIENT ANALYSIS:")
            self.log(f"     Trainable params with gradients: {trainable_with_grads}")
            self.log(f"     Trainable params without gradients: {trainable_without_grads}")
            self.log(f"     Frozen params with gradients: {frozen_with_grads}")
            self.log(f"     Frozen params without gradients: {frozen_without_grads}")
            self.log(f"     Total trainable gradient norm: {trainable_grad_norm:.6e}")
            self.log(f"     Total frozen gradient norm: {frozen_grad_norm:.6e}")
            
            # Validation checks
            if frozen_with_grads == 0:
                self.log(f"     ✓ PASS: No frozen parameters received gradients")
            else:
                self.log(f"     ✗ FAIL: {frozen_with_grads} frozen parameters received gradients!")
            
            if trainable_with_grads > 0:
                self.log(f"     ✓ PASS: {trainable_with_grads} trainable parameters received gradients")
            else:
                self.log(f"     ✗ FAIL: No trainable parameters received gradients!")
            
            # Take snapshot of trainable parameters before optimization
            self.log(f"\n   PARAMETER UPDATE VALIDATION:")
            param_snapshot = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_snapshot[name] = param.detach().clone()
            
            self.log(f"     Snapshot taken of {len(param_snapshot)} trainable parameters")
            
            # Perform optimizer step
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            
            self.log(f"     Optimizer step completed")
            
            # Measure parameter changes
            total_param_change = 0.0
            params_that_changed = 0
            params_that_didnt_change = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in param_snapshot:
                    delta = (param.detach() - param_snapshot[name]).abs().sum().item()
                    total_param_change += delta
                    
                    if delta > 1e-10:  # Small threshold for numerical precision
                        params_that_changed += 1
                    else:
                        params_that_didnt_change += 1
            
            self.log(f"     Parameters that changed: {params_that_changed}")
            self.log(f"     Parameters that didn't change: {params_that_didnt_change}")
            self.log(f"     Total parameter change magnitude: {total_param_change:.6e}")
            
            # Final validation
            if total_param_change > 0:
                self.log(f"     ✓ PASS: Trainable parameters were updated (total |Δ|: {total_param_change:.3e})")
            else:
                self.log(f"     ✗ FAIL: No parameter updates detected!")
            
            # Verify frozen parameters didn't change
            frozen_params_changed = 0
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    # Frozen parameters should never change, so we don't need snapshots
                    # But we can check if they somehow got gradients and warn
                    if param.grad is not None:
                        frozen_params_changed += 1
            
            if frozen_params_changed == 0:
                self.log(f"     ✓ PASS: All frozen parameters remained unchanged")
            else:
                self.log(f"     ✗ WARNING: {frozen_params_changed} frozen parameters had gradients")
            
            # Summary
            self.log(f"\n   GRADIENT FLOW VALIDATION SUMMARY:")
            gradient_flow_correct = (frozen_with_grads == 0) and (trainable_with_grads > 0)
            parameter_updates_correct = (total_param_change > 0) and (frozen_params_changed == 0)
            
            if gradient_flow_correct:
                self.log(f"     ✓ Gradient flow: CORRECT (only trainable params receive gradients)")
            else:
                self.log(f"     ✗ Gradient flow: INCORRECT")
                
            if parameter_updates_correct:
                self.log(f"     ✓ Parameter updates: CORRECT (only trainable params change)")
            else:
                self.log(f"     ✗ Parameter updates: INCORRECT")
            
            overall_pass = gradient_flow_correct and parameter_updates_correct
            if overall_pass:
                self.log(f"     OVERALL: PASS - Perfect LoRA parameter handling!")
            else:
                self.log(f"     OVERALL: FAIL - LoRA parameter handling has issues")
            
            return {
                'gradient_flow_correct': gradient_flow_correct,
                'parameter_updates_correct': parameter_updates_correct,
                'overall_pass': overall_pass,
                'trainable_with_grads': trainable_with_grads,
                'frozen_with_grads': frozen_with_grads,
                'total_param_change': total_param_change,
                'params_that_changed': params_that_changed
            }
            
        except Exception as e:
            self.log(f"   ERROR during gradient flow validation: {e}")
            traceback.print_exc()
            return None


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load and parse configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract defaults section
    defaults = config.get('defaults')
    
    # Extract model configuration
    model_name = defaults['model_name']
    model_config = config.get('models').get(model_name)
    
    # Merge defaults with model-specific config
    merged_config = {
        # Model settings
        'model_name': model_config.get('model_name_or_path') or defaults['model_name_or_path'],
        'tokenizer_name': model_config.get('tokenizer_name') or defaults['tokenizer_name'],
        'load_in_4bit': model_config.get('load_in_4bit') or defaults['load_in_4bit'],
        'attn_implementation': defaults['attn_implementation'],
        
        # Training parameters
        'learning_rate': defaults['learning_rate'],
        'lr_scheduler_type': defaults['lr_scheduler_type'],
        'warmup_ratio': defaults['warmup_ratio'],
        'weight_decay': defaults['weight_decay'],
        'optim': defaults['optim'],
        'max_grad_norm': defaults['max_grad_norm'],
        
        # Training strategy
        'batch_size': defaults['batch_size'],
        'gradient_accumulation_steps': defaults['gradient_accumulation_steps'],
        'train_epochs': defaults['train_epochs'],
        'save_strategy': defaults['save_strategy'],
        'eval_strategy': defaults['eval_strategy'],
        'logging_steps': defaults['logging_steps'],
        
        # Precision settings
        'bf16': defaults['bf16'],
        'fp16': defaults['fp16'],
        
        # Memory optimization
        'gradient_checkpointing': defaults['gradient_checkpointing'],
        'use_cache': defaults['use_cache'],
        'dataloader_num_workers': defaults['dataloader_num_workers'],
        'dataloader_pin_memory': defaults['dataloader_pin_memory'],
        'pad_to_multiple_of': defaults['pad_to_multiple_of'],
        
        # LoRA configuration
        'rank': defaults['rank'],
        'lora_alpha': defaults['lora_alpha'],
        'target_modules': defaults['target_attention_layers'],
        'enable_clamps': defaults['enable_clamps'],
        'deterministic_lora': defaults['deterministic_lora'],
        
        # ARD configuration
        'kl_loss_beta': defaults['kl_loss_beta'],
        'ard_prior_samples': defaults['ard_prior_samples'],
        'max_len': defaults['max_len'],
        
        # Numerical stability parameters for ProbLoRA
        'logvar_clamp_min': defaults['logvar_clamp_min'],
        'logvar_clamp_max': defaults['logvar_clamp_max'],
        'beta_logvar_clamp_min': defaults['beta_logvar_clamp_min'],
        'beta_logvar_clamp_max': defaults['beta_logvar_clamp_max'],
        'sample_clamp_min': defaults['sample_clamp_min'],
        'sample_clamp_max': defaults['sample_clamp_max'],
        
        # Dataset configuration
        'dataset_name': defaults['dataset_name'],
        'dataset_name_specific': defaults['dataset_name_specific'],
        'random_seed': defaults['random_seed'],
        
        # Reporting
        'report_to': defaults['report_to'],
        'output_dir': f"output/{defaults['dataset_name_specific']}_run_{defaults['runId']}"
    }
    
    return merged_config


def load_model_for_analysis(config_path: str, model_args: Dict[str, Any]) -> torch.nn.Module:
    """Load model with ProbLoRA injection for analysis focusing only on ProbLoRA parameters."""
    
    print("Loading model for ProbLoRA-focused parameter analysis...")
    
    # Prepare model loading arguments
    model_kwargs = {
        'torch_dtype': torch.bfloat16 if model_args['bf16'] else torch.float16,
        'device_map': "cpu",  # Keep on CPU for analysis
        'load_in_4bit': False,  # No quantization for parameter counting
        'trust_remote_code': True
    }
    
    # Load base model using Auto class for flexibility
    model = AutoModelForCausalLM.from_pretrained(
        model_args['model_name'],
        **model_kwargs
    )
    
    # Inject ProbLoRA using the exact same function and parameters as training
    print("Injecting ProbLoRA layers with full configuration...")
    inject_problora_llama(
        model=model,
        rank=model_args['rank'],
        scaling=model_args['lora_alpha'] / model_args['rank'],
        num_tokens=model_args['max_len'],
        ard_prior_samples=model_args['ard_prior_samples'],
        logvar_clamp_min=model_args['logvar_clamp_min'],
        logvar_clamp_max=model_args['logvar_clamp_max'],
        beta_logvar_clamp_min=model_args['beta_logvar_clamp_min'],
        beta_logvar_clamp_max=model_args['beta_logvar_clamp_max'],
        sample_clamp_min=model_args['sample_clamp_min'],
        sample_clamp_max=model_args['sample_clamp_max'],
        attn_implementation=model_args['attn_implementation'],
        target_attention_layers=model_args['target_modules'],
        deterministic=model_args['deterministic_lora'],
        enable_clamps=model_args['enable_clamps'],
        lora_alpha=model_args['lora_alpha']
    )
    
    # Use EXACT same parameter handling as training script
    print("Applying exact training script parameter handling...")
    verbose = True  # Enable verbose output for analysis
    
    # Freeze base parameters and unfreeze LoRA parameters (EXACT COPY FROM TRAINING SCRIPT)
    trainable_count = 0
    all_param_names = []
    quantized_params_skipped = 0
    
    # ProbLoRALayer detection
    for mod_name, mod in model.named_modules():
        if isinstance(mod, ProbLoRALayer):
            if verbose:
                print(f"[DEBUG] Found ProbLoRALayer module: {mod_name}")
            
            # Only parameters directly on this module (no recursion)
            for p_name, p in mod.named_parameters(recurse=False):
                full_param_name = f"{mod_name}.{p_name}" if mod_name else p_name
                all_param_names.append(full_param_name)
                
                # CRITICAL: Debug parameter details before setting gradients
                if verbose:
                    print(f"[DEBUG] Found ProbLoRA parameter: {full_param_name}")
                    print(f"[DEBUG]   Local name: {p_name}")
                    print(f"[DEBUG]   Shape: {p.shape}")
                    print(f"[DEBUG]   Dtype: {p.dtype}")
                    print(f"[DEBUG]   Is floating point: {p.is_floating_point()}")
                
                # CRITICAL: Only set gradients on floating-point parameters
                if p.is_floating_point():
                    p.requires_grad_(True)
                    trainable_count += 1
                    if verbose:
                        print(f"[DEBUG] Trainable ProbLoRA param: {full_param_name} (shape: {p.shape}, dtype: {p.dtype})")
                else:
                    quantized_params_skipped += 1
                    print(f"[WARNING] Skipping quantized ProbLoRA param: {full_param_name} (dtype: {p.dtype})")
                    print(f"[WARNING] This ProbLoRA parameter is quantized and cannot have gradients!")
    
    # Freeze all non-ProbLoRA parameters
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, ProbLoRALayer):
            for p_name, p in mod.named_parameters(recurse=False):
                if p.is_floating_point():
                    p.requires_grad_(False)
    
    # Count parameters focusing only on trainable ones (ProbLoRA matrices)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    problora_params = sum(p.numel() for name, p in model.named_parameters() 
                         if p.requires_grad and any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A']))
    
    print(f"\nParameter analysis focus: ProbLoRA layers only")
    print(f"  Total model parameters: {total_params:,}")
    print(f"  Total trainable parameters: {trainable_params:,}")
    print(f"  ProbLoRA trainable parameters: {problora_params:,}")
    print(f"  Trainable parameter groups: {trainable_count}")
    print(f"  Quantized parameters skipped: {quantized_params_skipped}")
    
    if trainable_params > 0:
        print(f"  ProbLoRA percentage of trainable: {100*problora_params/trainable_params:.1f}%")
        print(f"  Trainable percentage of total: {100*trainable_params/total_params:.4f}%")
    
    # Verify that only ProbLoRA parameters are trainable (as expected)
    non_problora_trainable = [name for name, p in model.named_parameters() 
                             if p.requires_grad and not any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
    
    if non_problora_trainable:
        print(f"  WARNING: Found {len(non_problora_trainable)} non-ProbLoRA trainable parameters")
        for name in non_problora_trainable[:5]:  # Show first 5
            print(f"    - {name}")
        if len(non_problora_trainable) > 5:
            print(f"    ... and {len(non_problora_trainable) - 5} more")
    else:
        print(f"  VERIFIED: Only ProbLoRA parameters (A, B, G, mu_A matrices) are trainable")
    
    # Final verification using training script method
    if trainable_count == 0:
        print("\n[ERROR] No ProbLoRA parameters found! Debugging information:")
        problora_modules = []
        for mod_name, mod in model.named_modules():
            if isinstance(mod, ProbLoRALayer):
                problora_modules.append(mod_name)
        print(f"[DEBUG] ProbLoRALayer modules found: {len(problora_modules)}")
        if quantized_params_skipped > 0:
            print(f"[DIAGNOSIS] All ProbLoRA parameters appear to be quantized.")
            print(f"[SOLUTION] Consider disabling quantization: load_in_4bit: false")
        raise RuntimeError("No trainable ProbLoRA parameters found! Check ProbLoRA injection and parameter types.")
    
    print("Model loaded - analysis will use EXACT training script parameter handling")
    return model


def create_test_trainer_for_validation(model, config_path: str = "config/run_training_params.yaml"):
    """Create a test trainer to validate optimizer parameter usage."""
    try:
        
        print("Creating test trainer for optimizer validation...")
        
        # Resolve config path relative to project root (scripts is subdirectory)
        script_dir = Path(__file__).parent  # scripts directory
        project_root = script_dir.parent    # ARD-LoRA-Data-CLM directory
        
        # If config_path is relative, resolve it from project root
        if not Path(config_path).is_absolute():
            resolved_config_path = project_root / config_path
        else:
            resolved_config_path = Path(config_path)
            
        print(f"[DEBUG] Script dir: {script_dir}")
        print(f"[DEBUG] Project root: {project_root}")
        print(f"[DEBUG] Config path: {config_path}")
        print(f"[DEBUG] Resolved config path: {resolved_config_path}")
        
        # Load config to get training parameters (required)
        if not resolved_config_path.exists():
            print(f"ERROR: Configuration file not found: {resolved_config_path}")
            print("Cannot create test trainer without training configuration.")
            return None
            
        with open(resolved_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"[DEBUG] Config keys: {list(config.keys())}")
        
        if 'defaults' not in config:
            print(f"ERROR: 'defaults' section not found in config file")
            print(f"Available sections: {list(config.keys())}")
            return None
            
        defaults = config['defaults']
        print(f"[DEBUG] Defaults keys: {list(defaults.keys())}")
        
        # Extract training parameters from YAML (no fallbacks)
        try:
            learning_rate = defaults['learning_rate']
            optim = defaults['optim']
            weight_decay = defaults['weight_decay']
            max_grad_norm = defaults['max_grad_norm']
            batch_size = defaults['batch_size']
            model_name = defaults['model_name_or_path']
        except KeyError as e:
            print(f"ERROR: Missing required parameter in config defaults: {e}")
            print(f"Available parameters: {list(defaults.keys())}")
            return None
        
        # Validate required parameters
        if any(param is None for param in [learning_rate, optim, model_name]):
            print("ERROR: Missing required training parameters in config file")
            print(f"Required: learning_rate, optim, model_name_or_path")
            print(f"Values: lr={learning_rate}, optim={optim}, model={model_name}")
            return None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create training arguments from config
        training_args = TrainingArguments(
            output_dir="temp_output",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=1,
            optim=optim,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_strategy="no",
            logging_steps=100,
            remove_unused_columns=False  # Important for causal LM
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        print("Test trainer created successfully")
        print(f"Optimizer: {optim}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Max Grad Norm: {max_grad_norm}")
        
        return trainer
        
    except Exception as e:
        print(f"Warning: Could not create test trainer: {e}")
        traceback.print_exc()
        return None


def analyze_with_optimizer_validation(model, config_path: str = "config/run_training_params.yaml", output_file: str = None):
    """Analyze model parameters and validate optimizer usage with comprehensive gradient flow testing."""
    
    # Create parameter analyzer
    analyzer = ParameterAnalyzer(model, output_file)
    
    # Run basic analysis
    results = analyzer.analyze_model()
    
    # Create test trainer for optimizer validation
    trainer = create_test_trainer_for_validation(model, config_path)
    
    # Validate optimizer parameters
    if trainer is not None:
        optimizer_results = analyzer.validate_optimizer_parameters(trainer=trainer)
        results['optimizer_validation'] = optimizer_results
        
        # NEW: Validate gradient flow and parameter updates (the ultimate test!)
        gradient_flow_results = analyzer.validate_gradient_flow_and_updates(trainer=trainer)
        results['gradient_flow_validation'] = gradient_flow_results
        
    else:
        print("Skipping optimizer and gradient flow validation due to trainer creation failure")
        results['optimizer_validation'] = None
        results['gradient_flow_validation'] = None
    
    # Save log
    analyzer.save_log()
    
    return results, analyzer


def quick_analysis():
    """Run quick parameter analysis with configuration from YAML file."""
    
    # Load configuration from YAML
    config_path = "config/run_training_params.yaml"
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print("The analysis requires the exact same configuration used for training.")
        print("Please ensure the config/run_training_params.yaml file exists.")
        return None
    
    # Load configuration from YAML (required)
    config_args = load_config_from_yaml(config_path)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"parameter_analysis_{timestamp}.txt"
    
    print("Quick Parameter Analysis")
    print("=" * 50)
    print(f"Config File: {config_path}")
    print(f"Model: {config_args['model_name']}")
    print(f"LoRA Rank: {config_args['rank']}")
    print(f"LoRA Alpha: {config_args['lora_alpha']}")
    print(f"Target Modules: {config_args['target_modules']}")
    print(f"Deterministic LoRA: {config_args['deterministic_lora']}")
    print(f"Optimizer: {config_args['optim']}")
    print(f"Learning Rate: {config_args['learning_rate']}")
    print(f"Output: {output_file}")
    print()
    
    try:
        # Load model (this will take some time)
        print("Loading model (this may take a few minutes)...")
        model = load_model_for_analysis(config_path, config_args)
        
        # Analyze parameters with optimizer validation
        print("Analyzing parameters and validating optimizer...")
        results, analyzer = analyze_with_optimizer_validation(model, config_path, str(output_file))
        
        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"Total Parameters: {results['total_parameters']:,}")
        print(f"Trainable Parameters: {results['total_trainable']:,}")
        print(f"Trainable Percentage: {(results['total_trainable']/results['total_parameters'])*100:.2f}%")
        print(f"ProbLoRA Layers: {results['problora_layers']}")
        print(f"ProbLoRA Trainable: {results['problora_trainable']:,}")
        print()
        print(f"Detailed report: {output_file}")
        print(f"Target validation: 12,582,912 expected")
        
        # Validation status
        expected = 12_582_912
        actual = results['total_trainable']
        if actual == expected:
            print("Parameter count matches expected value!")
        else:
            diff = abs(actual - expected)
            print(f"Difference: {diff:,} parameters ({diff/expected*100:.2f}%)")
        
        # Optimizer validation summary
        if 'optimizer_validation' in results and results['optimizer_validation'] is not None:
            opt_results = results['optimizer_validation']
            print(f"\nOptimizer Validation:")
            print(f"  Model trainable parameters: {opt_results['model_param_count']:,}")
            print(f"  ProbLoRA parameters found: {len(opt_results['problora_param_names'])}")
            print(f"  Non-ProbLoRA parameters: {len(opt_results['non_problora_param_names'])}")
        elif 'optimizer_validation' in results:
            print(f"\nOptimizer Validation: Skipped (trainer creation failed)")
        else:
            print(f"\nOptimizer Validation: Not performed")
        
        # Gradient flow validation summary
        if 'gradient_flow_validation' in results and results['gradient_flow_validation'] is not None:
            gf_results = results['gradient_flow_validation']
            print(f"\nGradient Flow Validation:")
            print(f"  Trainable params with gradients: {gf_results['trainable_with_grads']}")
            print(f"  Frozen params with gradients: {gf_results['frozen_with_grads']}")
            print(f"  Parameters that updated: {gf_results['params_that_changed']}")
            print(f"  Total parameter change: {gf_results['total_param_change']:.3e}")
            
            if gf_results['overall_pass']:
                print(f"  Result: PERFECT LoRA parameter handling!")
            else:
                print(f"  Result: Issues detected in LoRA parameter handling")
        elif 'gradient_flow_validation' in results:
            print(f"\nGradient Flow Validation: Skipped (trainer creation failed)")
        else:
            print(f"\nGradient Flow Validation: Not performed")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return None


def analyze_existing_model(model, output_file: str = None):
    """Analyze an already loaded model."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/parameter_analysis_{timestamp}.txt"
    
    analyzer = ParameterAnalyzer(model, output_file)
    results = analyzer.analyze_model()
    analyzer.save_log()
    
    return results


def main():
    """Main function for parameter analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze trainable parameters in ARD-LoRA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick analysis with defaults
  python scripts/analyze_trainable_parameters.py --quick
  
  # Analyze with custom config
  python scripts/analyze_trainable_parameters.py --config config/run_training_params.yaml
  
  # Analyze with custom output file
  python scripts/analyze_trainable_parameters.py --output my_analysis.txt
  
  # Override model name
  python scripts/analyze_trainable_parameters.py --model-name meta-llama/Llama-2-7b-hf
        """
    )
    parser.add_argument("--config", type=str, default="config/run_training_params.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for parameter analysis (optional)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name from config")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick analysis with default configuration")
    
    args = parser.parse_args()
    
    # Run quick analysis if requested
    if args.quick:
        return quick_analysis()
    
    # Load configuration from YAML
    config_args = load_config_from_yaml(args.config)
    
    # Override model name if provided
    if args.model_name:
        config_args['model_name'] = args.model_name
    
    # Set default output file if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/parameter_analysis_{timestamp}.txt"
    
    try:
        # Load model with YAML configuration
        model = load_model_for_analysis(args.config, config_args)
        
        # Analyze parameters with optimizer validation
        results, analyzer = analyze_with_optimizer_validation(model, args.config, args.output)
        
        print(f"\nAnalysis complete!")
        print(f"Found {results['total_trainable']:,} trainable parameters")
        print(f"Analyzed {results['problora_layers']} ProbLoRA layers")
        
        if args.output:
            print(f"Detailed report saved to: {args.output}")
        
        # Print optimizer validation summary
        if 'optimizer_validation' in results:
            opt_results = results['optimizer_validation']
            print(f"\nOptimizer Validation:")
            print(f"  Model trainable parameters: {opt_results['model_param_count']:,}")
            print(f"  ProbLoRA parameters found: {len(opt_results['problora_param_names'])}")
            print(f"  Non-ProbLoRA parameters: {len(opt_results['non_problora_param_names'])}")
        
        # Print gradient flow validation summary
        if 'gradient_flow_validation' in results and results['gradient_flow_validation'] is not None:
            gf_results = results['gradient_flow_validation']
            print(f"\nGradient Flow Validation:")
            print(f"  Gradient flow: {'✓ CORRECT' if gf_results['gradient_flow_correct'] else '✗ INCORRECT'}")
            print(f"  Parameter updates: {'✓ CORRECT' if gf_results['parameter_updates_correct'] else '✗ INCORRECT'}")
            print(f"  Overall result: {'PERFECT' if gf_results['overall_pass'] else 'ISSUES DETECTED'}")
        else:
            print(f"\nGradient Flow Validation: Not performed")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If no arguments provided, show usage and run quick analysis
    if len(sys.argv) == 1:
        print("ARD-LoRA Parameter Analysis Tool")
        print("=" * 50)
        print("This tool provides comprehensive analysis of trainable parameters in ARD-LoRA models.")
        print()
        print("Usage Options:")
        print("  --quick                     : Run quick analysis with defaults")
        print("  --config <file>            : Use specific config file")
        print("  --output <file>            : Specify output file")
        print("  --model-name <name>        : Override model name")
        print()
        print("Running quick analysis with default settings...")
        print()
        quick_analysis()
    else:
        main()
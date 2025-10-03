import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class ProbLoRALayer(nn.Module):
    """
    Probabilistic LoRA Layer with ARD priors - model-agnostic implementation.
    Works with both encoder (bidirectional) and decoder (causal) architectures.
    The KL divergence computation is independent of attention masking.
    """
    def __init__(self, base_proj: nn.Linear, rank, num_tokens=2048, ard_prior_samples=1000, scaling=1.0):
        super().__init__()

        self.base_proj = base_proj
        self.base_proj.requires_grad_(False)  # freeze base weights
        self.rank = rank
        self.num_tokens = num_tokens
        self.ard_prior_samples = ard_prior_samples
        self.scaling = scaling

        self.in_features = base_proj.in_features
        self.out_features = base_proj.out_features

        # # Init mu_A, logvar_A separately for stability
        # mu_A = torch.empty(rank, self.in_features)
        # nn.init.xavier_normal_(mu_A)
        # logvar_A = torch.full((rank, self.in_features), float(torch.log(torch.tensor(1e-2))))
        # self.A = nn.Parameter(torch.cat([mu_A, logvar_A], dim=0))   # Wrap as parameters

        # # Apply Xavier Normal initialization on matrix B
        # B_tensor = torch.empty(self.out_features, self.rank)
        # nn.init.xavier_normal_(B_tensor)
        # self.B = nn.Parameter(B_tensor)     # Wrap as parameters

        # Create raw tensors
        A_tensor = torch.empty(2*rank, self.in_features)
        B_tensor = torch.empty(self.out_features, self.rank)

        # Apply Xavier Normal initialization
        nn.init.xavier_normal_(A_tensor)
        nn.init.xavier_normal_(B_tensor)

        # Wrap as parameters
        self.A = nn.Parameter(A_tensor)
        self.B = nn.Parameter(B_tensor)

        # ARD prior parameter
        self.alpha = (self.num_tokens*self.ard_prior_samples) / 2.0
        self.beta = np.zeros(self.rank, dtype=np.float32)
        self.register_buffer('est_var', torch.ones(self.rank))  # Initialize est_var as a buffer (will move with model to device)

    def forward(self, x):
        """Forward pass - works with any sequence length and masking strategy"""
        base_out = self.base_proj(x)            # shape: [B, S, out_dim]
        mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        
        # Convert LoRA parameters to input dtype and device for computation (BF16/CUDA compatibility)
        # Keep original parameters in FP32 for precise gradient updates
        mu_A = mu_A.to(dtype=x.dtype, device=x.device)
        logvar_A = logvar_A.to(dtype=x.dtype, device=x.device)
        B_matrix = self.B.to(dtype=x.dtype, device=x.device)
        
        # Apply variance mask if it exists
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            # Apply mask to both mu_A and logvar_A (mask inactive latent dimensions)
            # mu_A and logvar_A shapes: [rank, in_features]
            mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
            mu_A_masked = mu_A * mask
            logvar_A_masked = logvar_A * mask
            # Also apply mask to B matrix - B shape: [out_features, rank]
            # Mask the rank dimensions (columns of B)
            B_masked = B_matrix * self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # Shape: [1, rank]
        else:
            # No mask, use original matrices
            mu_A_masked = mu_A
            logvar_A_masked = logvar_A
            B_masked = B_matrix
        
        B, S, _ = x.size()
        x_flat = x.view(-1, x.size(-1))  # [B*S, in_features]
        
        # Compute latent mean and logvar: [B*S, rank]
        mu = (mu_A_masked @ x_flat.T).T
        # logvar_A_masked = torch.log(F.softplus(logvar_A_masked) + 1e-6) # Stable gradient for variance
        logvar = (logvar_A_masked @ x_flat.T).T
        
        # Apply additional masking to latent outputs to ensure inactive dims are zero
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            mask_latent = self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # [1, rank]
            mu = mu * mask_latent
            logvar = logvar * mask_latent
        
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)  # [B*S, rank]
        
        # Final LoRA output: z @ B_masked^T
        out = (z @ B_masked.T).view(B, S, -1)     # shape: [B, S, out_dim]
        final_out = base_out + self.scaling * out

        return final_out

    def kl_divergence_latent(self, x):
        """
        Optimized KL divergence computation - model-agnostic.
        Works with both causal (LLaMA) and bidirectional (DeBERTa) attention.
        The computation is in the latent space and independent of masking strategy.
        
        Optimizations:
        - Reduced redundant matrix operations
        - Cached device/dtype conversions
        - Eliminated redundant exp() calls
        - More efficient tensor operations
        """
        mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        B, S, _ = x.size()  # batch_size, seq_len, features
        
        # Early exit for zero active dimensions
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            active_dims = torch.sum(self.variance_mask > 0).item()
            if active_dims == 0:
                return torch.tensor(0.0, device=x.device, requires_grad=True)
        else:
            active_dims = self.rank
        
        # Flatten input once
        x_flat = x.view(-1, x.size(-1))  # [B*S, in_features]
        
        # Convert to input dtype and device for computation consistency (cached)
        mu_A = mu_A.to(dtype=x.dtype, device=x.device)
        logvar_A = logvar_A.to(dtype=x.dtype, device=x.device)
        
        # Apply variance mask if it exists (optimized)
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
            mu_A_masked = mu_A * mask
            logvar_A_masked = logvar_A * mask
            
            # Optimized matrix operations for masked case
            mu = torch.mm(x_flat, mu_A_masked.T)  # [B*S, rank] - more efficient than transpose operations
            logvar = torch.mm(x_flat, logvar_A_masked.T)  # [B*S, rank]
            
            # Get target variance for active dimensions only
            active_mask = self.variance_mask > 0
            if hasattr(self, 'est_var') and self.est_var is not None:
                target_var = self.est_var[active_mask].to(x.device).unsqueeze(0)
            else:
                target_var = torch.ones(active_dims, device=x.device, dtype=x.dtype).unsqueeze(0)
            
            # Select only active dimensions for computation
            active_indices = torch.where(self.variance_mask > 0)[0]
            mu_active = mu[:, active_indices]
            logvar_active = logvar[:, active_indices]
            
            # Optimized KL computation: avoid redundant exp() call
            # KL = 0.5 * (log(target_var) - logvar + (exp(logvar) + mu²) / target_var - 1)
            mu_squared = mu_active.pow(2)
            exp_logvar = torch.exp(logvar_active)
            log_target_var = torch.log(target_var)
            
            kld = 0.5 * (log_target_var - logvar_active + 
                        (exp_logvar + mu_squared) / target_var - 1.0)
        else:
            # Optimized matrix operations for non-masked case
            mu = torch.mm(x_flat, mu_A.T)  # [B*S, rank] - more efficient
            logvar = torch.mm(x_flat, logvar_A.T)  # [B*S, rank]
            
            # Get target variance
            target_var = self.est_var.to(x.device).unsqueeze(0)
            
            # Optimized KL computation: avoid redundant exp() call
            mu_squared = mu.pow(2)
            exp_logvar = torch.exp(logvar)
            log_target_var = torch.log(target_var)
            
            kld = 0.5 * (log_target_var - logvar + 
                        (exp_logvar + mu_squared) / target_var - 1.0)
        
        # Return mean KL divergence
        return kld.mean()
    
    def beta_get_sample(self, x):
        """Sample from latent distribution for ARD prior estimation"""
        mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        
        # Convert to input dtype and device for computation consistency
        mu_A = mu_A.to(dtype=x.dtype, device=x.device)
        logvar_A = logvar_A.to(dtype=x.dtype, device=x.device)
        
        # Apply variance mask if it exists
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            # Mask the latent dimensions (output dims of A matrices)
            mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
            mu_A_masked = mu_A * mask
            logvar_A_masked = logvar_A * mask
        else:
            mu_A_masked = mu_A
            logvar_A_masked = logvar_A
            
        x_flat = x.view(-1, x.size(-1))
        mu = (mu_A_masked @ x_flat.T).T      # [B*S, rank]
        logvar = (logvar_A_masked @ x_flat.T).T  # [B*S, rank]
        
        # Apply additional masking to latent outputs
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            mask_latent = self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # [1, rank]
            mu = mu * mask_latent
            logvar = logvar * mask_latent
            
        eps = torch.randn_like(mu)
        samples = mu + eps * torch.exp(0.5 * logvar)
        
        # Convert to float32 before numpy conversion (BFloat16 not supported by numpy)
        return samples.float().cpu().detach().numpy()




def inject_problora_llama(model, rank=64, scaling=1.0, num_tokens=2048, ard_prior_samples=1000):
    """
    Inject ProbLoRA into LLaMA2-7B model.
    Targets the standard attention projections: q_proj, k_proj, v_proj, o_proj
    """
    print(f"[INFO] Injecting ProbLoRA into LLaMA2 model with rank={rank}")
    
    layers_modified = 0
    
    # LLaMA2 has model.layers (list of transformer blocks)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                
                # Wrap standard LLaMA attention projections
                if hasattr(attn, 'q_proj') and isinstance(attn.q_proj, nn.Linear):
                    attn.q_proj = ProbLoRALayer(attn.q_proj, rank, num_tokens, ard_prior_samples, scaling)
                    layers_modified += 1
                
                # if hasattr(attn, 'k_proj') and isinstance(attn.k_proj, nn.Linear):
                #     attn.k_proj = ProbLoRALayer(attn.k_proj, rank, num_tokens, ard_prior_samples, scaling)
                #     layers_modified += 1
                    
                if hasattr(attn, 'v_proj') and isinstance(attn.v_proj, nn.Linear):
                    attn.v_proj = ProbLoRALayer(attn.v_proj, rank, num_tokens, ard_prior_samples, scaling)
                    layers_modified += 1
                    
                # if hasattr(attn, 'o_proj') and isinstance(attn.o_proj, nn.Linear):
                #     attn.o_proj = ProbLoRALayer(attn.o_proj, rank, num_tokens, ard_prior_samples, scaling)
                #     layers_modified += 1
    
    print(f"[INFO] Successfully injected ProbLoRA into {layers_modified} linear layers")
    print(f"[INFO] Each layer now has ARD-enabled latent space with rank={rank}")
    print(f"[INFO] KL divergence will be computed in latent space (model-agnostic)")
    
    return model
import torch
import triton
import triton.language as tl

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Normalize x1 and x2
    x1_norm = torch.norm(x1, p=p_norm, dim=-1, keepdim=True)
    x2_norm = torch.norm(x2, p=p_norm, dim=-1, keepdim=True)
    
    # Avoid division by zero
    x1_norm = torch.clamp(x1_norm, min=eps_norm)
    x2_norm = torch.clamp(x2_norm, min=eps_norm)
    
    # Normalize the tensors
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm
    
    # Compute pairwise distance
    diff = x1_normalized.unsqueeze(1) - x2_normalized.unsqueeze(0)
    distance = torch.norm(diff, p=p_norm, dim=-1)
    
    # Add small epsilon to avoid zero distance
    distance = torch.clamp(distance, min=eps_distance)
    
    return distance
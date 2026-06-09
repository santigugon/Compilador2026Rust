import torch
import triton
import triton.language as tl

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Normalize x1 and x2
    x1_norm = torch.norm(x1, p=p_norm, dim=-1, keepdim=True)
    x2_norm = torch.norm(x2, p=p_norm, dim=-1, keepdim=True)
    x1_normalized = x1 / (x1_norm + eps_norm)
    x2_normalized = x2 / (x2_norm + eps_norm)
    
    # Compute pairwise distance
    # Using torch.cdist for pairwise distance computation
    distance = torch.cdist(x1_normalized, x2_normalized, p=p_norm, compute_mode='use_mm_for_euclid_dist')
    
    # Add epsilon to distance to avoid zero distances
    distance = distance + eps_distance
    
    return distance
import torch
import triton
import triton.language as tl

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    # Validate inputs
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Compute pairwise distances
    diff = x1 - x2
    if p_distance == 2.0:
        distances = torch.sum(diff * diff, dim=dim_norm, keepdim=True)
    else:
        distances = torch.sum(torch.abs(diff) ** p_distance, dim=dim_norm, keepdim=True)
    distances = torch.sqrt(distances + eps_distance)
    
    # Normalize distances
    if p_norm == 2:
        norms = torch.sum(distances * distances, dim=dim_norm, keepdim=True)
    else:
        norms = torch.sum(torch.abs(distances) ** p_norm, dim=dim_norm, keepdim=True)
    norms = torch.pow(norms + eps_norm, 1.0 / p_norm)
    
    # Avoid division by zero
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    
    # Return normalized distances
    result = distances / norms
    
    if not keepdim:
        result = result.squeeze(dim=dim_norm)
    
    return result
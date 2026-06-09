import torch
import triton
import triton.language as tl

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    # Ensure inputs have the same shape
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Compute pairwise distances
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    squared_diff = diff ** 2
    sum_squared = squared_diff.sum(dim=-1)
    distances = sum_squared ** (1.0 / p_distance)
    
    # Add epsilon to avoid division by zero
    distances = distances + eps_distance
    
    # Normalize along the specified dimension
    # First compute the norm for each vector
    if dim_norm == -1:
        dim_norm = len(x1.shape) - 1
    
    # Compute the norm for normalization
    norm_tensor = x1.norm(p=p_norm, dim=dim_norm, keepdim=True)
    norm_tensor = norm_tensor + eps_norm
    
    # Normalize the distances
    normalized_distances = distances / norm_tensor
    
    # Handle keepdim
    if not keepdim:
        # Remove the dimension that was normalized
        normalized_distances = normalized_distances.squeeze(dim_norm)
    
    return normalized_distances
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-06, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
#     pairwise_distance = torch.norm(x1 - x2, p=p_distance, dim=-1, keepdim=keepdim)
#     pairwise_distance = pairwise_distance + eps_distance
#     normed_distance = pairwise_distance / torch.norm(pairwise_distance, p=p_norm, dim=dim_norm, keepdim=True).clamp(min=eps_norm)
#     return normed_distance

def test_normalize_pairwise_distance():
    results = {}
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[1.0, 2.5], [2.5, 4.0]])
    
    # Compute the normalized pairwise distance
    results["test_case_1"] = normalize_pairwise_distance(x1, x2, p_distance=2.0, dim_norm=0)
    # Normalize along a different dimension
    results["test_case_2"] = normalize_pairwise_distance(x1, x2, p_distance=1.0, dim_norm=0)

    return results

test_results = test_normalize_pairwise_distance()

import torch
import triton
import triton.language as tl

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    assert dim_norm >= 0 and dim_norm < len(x1.shape), "dim_norm must be a valid dimension index"
    
    # Compute pairwise distances
    diff = x1[:, None, :] - x2[None, :, :]
    if p_distance == 2.0:
        distances = (diff ** 2).sum(dim=-1)
    else:
        distances = (diff.abs() ** p_distance).sum(dim=-1)
    distances = distances.sqrt() + eps_distance
    
    # Normalize along specified dimension
    if p_norm == 2:
        norms = (x1 ** 2).sum(dim=dim_norm, keepdim=True)
    else:
        norms = (x1.abs() ** p_norm).sum(dim=dim_norm, keepdim=True)
    norms = norms.sqrt() + eps_norm
    
    # Apply normalization
    result = distances / norms
    
    if not keepdim:
        result = result.squeeze(dim=dim_norm)
    
    return result
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

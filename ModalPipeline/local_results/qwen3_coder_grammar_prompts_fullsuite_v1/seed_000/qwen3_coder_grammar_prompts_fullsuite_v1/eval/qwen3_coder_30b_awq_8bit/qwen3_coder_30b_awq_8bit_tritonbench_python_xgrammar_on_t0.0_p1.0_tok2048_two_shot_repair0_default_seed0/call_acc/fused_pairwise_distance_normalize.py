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
##################################################################################################################################################



import torch

def test_fused_pairwise_distance_normalize():
    results = {}

    # Test case 1: Basic functionality with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = fused_pairwise_distance_normalize(x1, x2)

    # Test case 2: Different p_norm value
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = fused_pairwise_distance_normalize(x1, x2, p_norm=1.0)

    # Test case 3: Different eps_norm value
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = fused_pairwise_distance_normalize(x1, x2, eps_norm=1e-10)

    # Test case 4: keepdim=True
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_4"] = fused_pairwise_distance_normalize(x1, x2, keepdim=True)

    return results

test_results = test_fused_pairwise_distance_normalize()

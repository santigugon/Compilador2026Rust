import torch
import triton
import triton.language as tl

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Handle scalar output_size
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Apply adaptive average pooling
    pooled_x1 = torch.nn.functional.adaptive_avg_pool2d(x1, output_size)
    pooled_x2 = torch.nn.functional.adaptive_avg_pool2d(x2, output_size)
    
    # Flatten the pooled tensors for distance computation
    pooled_x1_flat = pooled_x1.view(pooled_x1.size(0), -1)
    pooled_x2_flat = pooled_x2.view(pooled_x2.size(0), -1)
    
    # Compute pairwise distance
    # Using torch operations for distance calculation since it's not straightforward to implement
    # with pure Triton for all norms
    diff = pooled_x1_flat.unsqueeze(1) - pooled_x2_flat.unsqueeze(0)
    if p == 2.0:
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)
    elif p == 1.0:
        dist = torch.sum(torch.abs(diff), dim=-1) + eps
    elif p == float('inf'):
        dist = torch.max(torch.abs(diff), dim=-1)[0] + eps
    else:
        dist = torch.sum(torch.abs(diff) ** p, dim=-1) ** (1.0 / p) + eps
    
    if keepdim:
        # Reshape to keep dimensions
        dist = dist.unsqueeze(-1)
    
    return dist
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float=2.0, eps: float=1e-06, keepdim: bool=False) -> torch.Tensor:
#     pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
#     pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
#     diff = pooled_x1 - pooled_x2
#     dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
#     return dist

def test_fused_pairwise_distance_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_1"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8))

    # Test case 2: Different output size
    x1 = torch.rand((2, 3, 64, 64), device='cuda')
    x2 = torch.rand((2, 3, 64, 64), device='cuda')
    results["test_case_2"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(16, 16))

    # Test case 3: Different norm degree
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_3"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), p=1.0)

    # Test case 4: Keep dimension
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_4"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), keepdim=True)

    return results

test_results = test_fused_pairwise_distance_adaptive_avg_pool2d()

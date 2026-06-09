import torch
import triton
import triton.language as tl

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Handle output_size as int or tuple
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Apply adaptive average pooling
    pooled_x1 = torch.nn.functional.adaptive_avg_pool2d(x1, output_size)
    pooled_x2 = torch.nn.functional.adaptive_avg_pool2d(x2, output_size)
    
    # Flatten the pooled tensors
    pooled_x1 = pooled_x1.view(pooled_x1.size(0), -1)
    pooled_x2 = pooled_x2.view(pooled_x2.size(0), -1)
    
    # Compute pairwise distance
    diff = pooled_x1.unsqueeze(1) - pooled_x2.unsqueeze(0)
    
    # Apply norm
    if p == 2.0:
        # Use squared norm for efficiency
        squared_diff = diff * diff
        sum_squared = squared_diff.sum(dim=-1)
        result = torch.sqrt(sum_squared + eps)
    elif p == 1.0:
        abs_diff = torch.abs(diff)
        sum_abs = abs_diff.sum(dim=-1)
        result = sum_abs + eps
    else:
        # General Lp norm
        diff_pow = torch.pow(torch.abs(diff), p)
        sum_pow = diff_pow.sum(dim=-1)
        result = torch.pow(sum_pow + eps, 1.0 / p)
    
    # Handle keepdim
    if keepdim:
        # Reshape result to match expected dimensions
        result = result.unsqueeze(-1)
    
    return result
import torch
import triton
import triton.language as tl

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Handle adaptive average pooling
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
        distance = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)
    elif p == 1.0:
        distance = torch.sum(torch.abs(diff), dim=-1) + eps
    elif p == float('inf'):
        distance = torch.max(torch.abs(diff), dim=-1)[0] + eps
    else:
        distance = torch.sum(torch.abs(diff) ** p, dim=-1) ** (1.0 / p) + eps
    
    if keepdim:
        # Reshape to keep dimensions
        distance = distance.unsqueeze(-1)
    
    return distance
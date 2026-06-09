import torch
import triton
import triton.language as tl

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    
    # Compute cosine similarity along dim=1
    # First, compute dot product
    dot_product = torch.sum(x1 * x2, dim=1)
    
    # Compute norms
    norm_x1 = torch.norm(x1, p=2, dim=1)
    norm_x2 = torch.norm(x2, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (norm_x1 * norm_x2 + eps)
    
    # Add singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)
    
    # Apply 2D average pooling
    result = torch.nn.functional.avg_pool2d(
        cosine_sim, 
        kernel_size=(kernel_size, 1), 
        stride=(stride, 1), 
        padding=(padding, 0)
    )
    
    return result
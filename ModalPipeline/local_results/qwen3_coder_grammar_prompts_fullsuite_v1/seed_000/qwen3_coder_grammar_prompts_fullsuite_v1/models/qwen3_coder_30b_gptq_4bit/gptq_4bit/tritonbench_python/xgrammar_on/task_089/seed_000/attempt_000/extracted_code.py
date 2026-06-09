import torch
import triton
import triton.language as tl

def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    if stride is None:
        stride = kernel_size
    
    # Compute cosine similarity along dim=1
    # First compute dot product
    dot_product = torch.sum(x1 * x2, dim=1)
    # Compute norms
    x1_norm = torch.norm(x1, dim=1)
    x2_norm = torch.norm(x2, dim=1)
    # Compute cosine similarity
    cosine_sim = dot_product / (x1_norm * x2_norm + eps)
    
    # Add singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)
    
    # Apply 2D average pooling
    # For simplicity, we'll use PyTorch's avg_pool2d
    # but we need to ensure the tensor is properly shaped
    # The input to avg_pool2d should be (N, C, H, W)
    # cosine_sim is (N, 1, L) where L is the length after dim=1 reduction
    # We need to reshape to (N, 1, 1, L) to make it 2D
    if cosine_sim.dim() == 3:
        cosine_sim = cosine_sim.unsqueeze(2)  # (N, 1, 1, L)
    
    # Apply average pooling
    out = torch.nn.functional.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)
    
    return out
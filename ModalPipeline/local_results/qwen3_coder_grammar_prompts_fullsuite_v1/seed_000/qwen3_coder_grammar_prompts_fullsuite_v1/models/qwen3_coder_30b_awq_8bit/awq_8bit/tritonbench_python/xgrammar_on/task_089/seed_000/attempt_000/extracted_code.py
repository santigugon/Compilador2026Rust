import torch
import triton
import triton.language as tl

@triton.jit
def _cosine_similarity_kernel(x1_ptr, x2_ptr, out_ptr, n: tl.constexpr, dim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x1_vals = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2_vals = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product
    dot = x1_vals * x2_vals
    
    # Compute norms
    x1_norm = x1_vals * x1_vals
    x2_norm = x2_vals * x2_vals
    
    # Reduce over dimension
    dot_sum = tl.sum(dot, axis=0)
    x1_norm_sum = tl.sum(x1_norm, axis=0)
    x2_norm_sum = tl.sum(x2_norm, axis=0)
    
    # Compute cosine similarity
    norm_product = tl.sqrt(x1_norm_sum) * tl.sqrt(x2_norm_sum)
    cosine_sim = dot_sum / (norm_product + 1e-8)
    
    tl.store(out_ptr + offsets, cosine_sim, mask=mask)

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    # Compute cosine similarity along dim=1
    # For simplicity, we'll use PyTorch's cosine_similarity
    # and then manually implement the rest
    
    # Cosine similarity along dim=1
    # Assuming x1 and x2 have shape [N, D]
    # cosine_sim = sum(x1 * x2) / (sqrt(sum(x1^2)) * sqrt(sum(x2^2)))
    
    # Compute cosine similarity manually
    x1_norm = torch.norm(x1, dim=1, keepdim=True)
    x2_norm = torch.norm(x2, dim=1, keepdim=True)
    
    # Avoid division by zero
    x1_norm = torch.clamp(x1_norm, min=eps)
    x2_norm = torch.clamp(x2_norm, min=eps)
    
    # Compute dot product
    dot_product = torch.sum(x1 * x2, dim=1, keepdim=True)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (x1_norm * x2_norm)
    
    # Add singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)  # Shape: [N, 1, D]
    
    # Apply 2D average pooling
    # We need to reshape to [N, 1, D] and apply avg_pool2d
    # But avg_pool2d expects [N, C, H, W]
    # So we reshape to [N, 1, 1, D] and apply 2D pooling
    
    # Reshape to [N, 1, 1, D]
    pooled = torch.nn.functional.avg_pool2d(
        cosine_sim.unsqueeze(2),  # [N, 1, 1, D]
        kernel_size=(1, kernel_size),
        stride=(1, stride if stride is not None else kernel_size),
        padding=(0, padding)
    )
    
    return pooled
import torch
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_row_stride, x2_row_stride, out_row_stride,
    n_features, n_samples,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_samples:
        return
    
    x1_row = x1_ptr + row * x1_row_stride
    x2_row = x2_ptr + row * x2_row_stride
    
    # Compute dot product
    dot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x1_norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x2_norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(0, n_features, BLOCK_SIZE):
        mask = tl.arange(0, BLOCK_SIZE) + i < n_features
        x1_vals = tl.load(x1_row + i + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        x2_vals = tl.load(x2_row + i + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        
        dot += x1_vals * x2_vals
        x1_norm += x1_vals * x1_vals
        x2_norm += x2_vals * x2_vals
    
    dot = tl.sum(dot)
    x1_norm = tl.sum(x1_norm)
    x2_norm = tl.sum(x2_norm)
    
    # Add epsilon to prevent division by zero
    x1_norm = tl.maximum(x1_norm, eps)
    x2_norm = tl.maximum(x2_norm, eps)
    
    # Compute cosine similarity
    norm = tl.sqrt(x1_norm) * tl.sqrt(x2_norm)
    similarity = dot / norm
    
    # Write result
    out = out_ptr + row * out_row_stride
    tl.store(out, similarity)

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    
    # Compute cosine similarity along dim=1
    n_samples, n_features = x1.shape
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Allocate output for cosine similarity
    cos_sim = torch.empty(n_samples, dtype=torch.float32, device=x1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_samples, 1),)
    
    cosine_similarity_kernel[grid](
        x1, x2, cos_sim,
        x1.stride(0), x2.stride(0), cos_sim.stride(0),
        n_features, n_samples,
        eps,
        BLOCK_SIZE
    )
    
    # Add singleton dimension (unsqueeze)
    cos_sim = cos_sim.unsqueeze(1)  # Shape: [n_samples, 1]
    
    # Apply 2D average pooling
    # Reshape to 2D: [n_samples, 1, 1, 1] for pooling
    cos_sim = cos_sim.unsqueeze(2).unsqueeze(3)  # Shape: [n_samples, 1, 1, 1]
    
    # Apply average pooling
    pooled = torch.nn.functional.avg_pool2d(
        cos_sim,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding)
    )
    
    # Remove singleton dimensions
    result = pooled.squeeze(2).squeeze(2)
    
    return result

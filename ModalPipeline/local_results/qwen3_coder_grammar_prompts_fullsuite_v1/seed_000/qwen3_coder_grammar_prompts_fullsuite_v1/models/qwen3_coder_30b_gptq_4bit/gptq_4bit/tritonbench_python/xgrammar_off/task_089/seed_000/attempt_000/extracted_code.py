import torch
import triton
import triton.language as tl

@triton.jit
def _cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    n_features: tl.constexpr,
    n_samples: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_samples
    
    # Load x1 and x2
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product
    dot = tl.sum(x1 * x2)
    
    # Compute norms
    norm_x1 = tl.sqrt(tl.sum(x1 * x1) + eps)
    norm_x2 = tl.sqrt(tl.sum(x2 * x2) + eps)
    
    # Compute cosine similarity
    cosine_sim = dot / (norm_x1 * norm_x2)
    
    # Store result
    tl.store(out_ptr + pid, cosine_sim, mask=mask)

def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    # Compute cosine similarity along dim=1
    n_samples, n_features = x1.shape
    
    # Create output tensor for cosine similarity
    cosine_sim_out = torch.empty(n_samples, dtype=x1.dtype, device=x1.device)
    
    # Compute cosine similarity using Triton kernel
    block = 256
    grid = triton.cdiv(n_samples, block)
    _cosine_similarity_kernel[grid](
        x1, x2, cosine_sim_out,
        n_features, n_samples, eps, BLOCK=block
    )
    
    # Add singleton dimension using unsqueeze
    cosine_sim_reshaped = cosine_sim_out.unsqueeze(1)
    
    # Apply 2D average pooling
    # We need to reshape to 2D for pooling
    # Assuming we want to pool along the feature dimension
    # Reshape to (batch, 1, 1, features) for pooling
    batch_size = cosine_sim_reshaped.shape[0]
    pooled = torch.nn.functional.avg_pool2d(
        cosine_sim_reshaped.view(batch_size, 1, 1, -1),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    
    # Reshape back to original dimensions
    return pooled.view(batch_size, -1)

import torch
import triton
import triton.language as tl

def _get_norm(x, dim, p_norm, eps_norm):
    # Compute L_p norm along specified dimension
    if p_norm == 2:
        x_norm = tl.sqrt(tl.sum(x * x, axis=dim, keepdims=True) + eps_norm)
    elif p_norm == 1:
        x_norm = tl.sum(tl.abs(x), axis=dim, keepdims=True) + eps_norm
    else:
        x_norm = tl.pow(tl.sum(tl.pow(tl.abs(x), p_norm), axis=dim, keepdims=True) + eps_norm, 1.0 / p_norm)
    return x_norm

@triton.jit
def _normalized_cosine_similarity_kernel(x1_ptr, x2_ptr, out_ptr, x1_stride_0, x1_stride_1, x2_stride_0, x2_stride_1, out_stride_0, out_stride_1, n_elements, dim: tl.constexpr, p_norm: tl.constexpr, eps_similarity: tl.constexpr, eps_norm: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Compute offsets for the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x1 and x2
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute norms
    x1_norm = _get_norm(x1, dim, p_norm, eps_norm)
    x2_norm = _get_norm(x2, dim, p_norm, eps_norm)
    
    # Normalize
    x1_normed = x1 / x1_norm
    x2_normed = x2 / x2_norm
    
    # Compute dot product
    dot_product = tl.sum(x1_normed * x2_normed, axis=dim, keepdims=True)
    
    # Compute cosine similarity
    similarity = dot_product / (tl.sqrt(tl.sum(x1_normed * x1_normed, axis=dim, keepdims=True) + eps_similarity) * 
                               tl.sqrt(tl.sum(x2_normed * x2_normed, axis=dim, keepdims=True) + eps_similarity))
    
    # Store result
    tl.store(out_ptr + offsets, similarity, mask=mask)

@triton.jit
def _normalized_cosine_similarity_kernel_2d(x1_ptr, x2_ptr, out_ptr, x1_stride_0, x1_stride_1, x2_stride_0, x2_stride_1, out_stride_0, out_stride_1, n_elements, dim: tl.constexpr, p_norm: tl.constexpr, eps_similarity: tl.constexpr, eps_norm: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Compute offsets for the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x1 and x2
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute norms
    x1_norm = _get_norm(x1, dim, p_norm, eps_norm)
    x2_norm = _get_norm(x2, dim, p_norm, eps_norm)
    
    # Normalize
    x1_normed = x1 / x1_norm
    x2_normed = x2 / x2_norm
    
    # Compute dot product
    dot_product = tl.sum(x1_normed * x2_normed, axis=dim, keepdims=True)
    
    # Compute cosine similarity
    similarity = dot_product / (tl.sqrt(tl.sum(x1_normed * x1_normed, axis=dim, keepdims=True) + eps_similarity) * 
                               tl.sqrt(tl.sum(x2_normed * x2_normed, axis=dim, keepdims=True) + eps_similarity))
    
    # Store result
    tl.store(out_ptr + offsets, similarity, mask=mask)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure inputs are contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Get output shape
    out_shape = list(x1.shape)
    out_shape.pop(dim)
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=torch.float32, device=x1.device)
    
    # Flatten tensors for processing
    x1_flat = x1.view(-1, x1.shape[dim])
    x2_flat = x2.view(-1, x2.shape[dim])
    out_flat = out.view(-1)
    
    # Get number of elements
    n_elements = out_flat.numel()
    
    # Define block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _normalized_cosine_similarity_kernel_2d[grid](
        x1_flat, x2_flat, out_flat,
        x1_flat.stride(0), x1_flat.stride(1),
        x2_flat.stride(0), x2_flat.stride(1),
        out_flat.stride(0), out_flat.stride(1),
        n_elements,
        dim,
        p_norm,
        eps_similarity,
        eps_norm,
        BLOCK_SIZE
    )
    
    return out
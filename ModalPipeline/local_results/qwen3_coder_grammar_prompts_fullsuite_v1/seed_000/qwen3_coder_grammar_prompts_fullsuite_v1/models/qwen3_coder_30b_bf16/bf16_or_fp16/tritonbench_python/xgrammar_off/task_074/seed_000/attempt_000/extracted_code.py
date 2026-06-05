import torch
import triton
import triton.language as tl

@triton.jit
def _norm_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x, stride_out, p_norm: tl.constexpr, eps_norm: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // dim_size
    dim_idx = pid % dim_size
    
    # Calculate the offset for this element
    offset = batch_idx * stride_x + dim_idx
    
    # Load the element
    x = tl.load(x_ptr + offset, mask=True)
    
    # Compute L_p norm
    if p_norm == 2:
        x_norm = tl.sqrt(x * x + eps_norm)
    elif p_norm == 1:
        x_norm = tl.abs(x) + eps_norm
    else:
        x_norm = tl.pow(tl.abs(x) + eps_norm, p_norm)
    
    # Store the result
    tl.store(out_ptr + offset, x_norm)

@triton.jit
def _cosine_similarity_kernel(x1_ptr, x2_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x1, stride_x2, stride_out, eps_similarity: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // dim_size
    dim_idx = pid % dim_size
    
    # Calculate the offset for this element
    offset = batch_idx * stride_x1 + dim_idx
    
    # Load the elements
    x1_val = tl.load(x1_ptr + offset, mask=True)
    x2_val = tl.load(x2_ptr + offset, mask=True)
    
    # Compute cosine similarity
    dot_product = x1_val * x2_val
    norm_product = tl.sqrt(tl.abs(x1_val) + 1e-12) * tl.sqrt(tl.abs(x2_val) + 1e-12)
    
    # Avoid division by zero
    similarity = dot_product / (norm_product + eps_similarity)
    
    # Store the result
    tl.store(out_ptr + offset, similarity)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure inputs are contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Get the shape and compute the total number of elements
    shape = x1.shape
    n = x1.numel()
    
    # Compute the size of the specified dimension
    dim_size = shape[dim]
    
    # Create output tensor
    out = torch.empty_like(x1)
    
    # Compute the number of batches
    batch_size = 1
    for i in range(len(shape)):
        if i != dim:
            batch_size *= shape[i]
    
    # Compute the stride for the specified dimension
    stride_x1 = x1.stride(dim)
    stride_x2 = x2.stride(dim)
    stride_out = out.stride(dim)
    
    # Launch kernel for normalization
    block = 256
    grid = (batch_size * dim_size,)
    
    # Normalize the tensors
    x1_norm = torch.empty_like(x1)
    x2_norm = torch.empty_like(x2)
    
    # Compute norms
    _norm_kernel[grid](x1, x1_norm, n, dim_size, stride_x1, stride_x1, p_norm, eps_norm, BLOCK=block)
    _norm_kernel[grid](x2, x2_norm, n, dim_size, stride_x2, stride_x2, p_norm, eps_norm, BLOCK=block)
    
    # Normalize the tensors
    x1_normalized = x1 / (x1_norm + eps_norm)
    x2_normalized = x2 / (x2_norm + eps_norm)
    
    # Compute cosine similarity
    _cosine_similarity_kernel[grid](x1_normalized, x2_normalized, out, n, dim_size, stride_x1, stride_x2, stride_out, eps_similarity, BLOCK=block)
    
    return out

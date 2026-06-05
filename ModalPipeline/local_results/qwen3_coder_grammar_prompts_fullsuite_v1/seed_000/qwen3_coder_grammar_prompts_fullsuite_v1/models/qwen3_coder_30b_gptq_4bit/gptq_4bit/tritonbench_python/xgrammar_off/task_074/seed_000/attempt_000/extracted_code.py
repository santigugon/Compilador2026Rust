import torch
import triton
import triton.language as tl

@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_stride, x2_stride, out_stride,
    dim_size, other_size,
    eps_similarity: tl.constexpr,
    eps_norm: tl.constexpr,
    p_norm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    # Calculate the ending position for this block
    end_pos = min(start_pos + BLOCK_SIZE, other_size)
    
    # Initialize accumulator for dot product
    dot_prod = tl.zeros([1], dtype=tl.float32)
    # Initialize accumulators for norms
    norm1 = tl.zeros([1], dtype=tl.float32)
    norm2 = tl.zeros([1], dtype=tl.float32)
    
    # Loop over the elements in the block
    for i in range(start_pos, end_pos):
        # Load elements from x1 and x2
        x1_val = tl.load(x1_ptr + i * x1_stride)
        x2_val = tl.load(x2_ptr + i * x2_stride)
        
        # Compute dot product
        dot_prod += x1_val * x2_val
        
        # Compute norms
        norm1 += x1_val * x1_val
        norm2 += x2_val * x2_val
    
    # Apply L_p normalization
    norm1 = tl.sqrt(norm1 + eps_norm)
    norm2 = tl.sqrt(norm2 + eps_norm)
    
    # Compute cosine similarity
    similarity = dot_prod / (norm1 * norm2 + eps_similarity)
    
    # Store result
    tl.store(out_ptr + block_idx * out_stride, similarity)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure inputs are on the same device and have the same dtype
    assert x1.device == x2.device, "x1 and x2 must be on the same device"
    assert x1.dtype == x2.dtype, "x1 and x2 must have the same dtype"
    
    # Ensure inputs have the same shape
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Ensure dim is valid
    assert dim >= 0 and dim < len(x1.shape), "dim must be valid"
    
    # Flatten the tensors along the specified dimension
    # Get the size of the specified dimension
    dim_size = x1.shape[dim]
    # Get the size of the other dimensions
    other_size = 1
    for i in range(len(x1.shape)):
        if i != dim:
            other_size *= x1.shape[i]
    
    # Reshape tensors to 1D for processing
    x1_flat = x1.view(-1)
    x2_flat = x2.view(-1)
    
    # Create output tensor
    out = torch.empty(other_size, dtype=torch.float32, device=x1.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (other_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _normalized_cosine_similarity_kernel[grid](
        x1_flat, x2_flat, out,
        x1_flat.stride(0), x2_flat.stride(0), out.stride(0),
        dim_size, other_size,
        eps_similarity,
        eps_norm,
        p_norm,
        BLOCK_SIZE
    )
    
    return out

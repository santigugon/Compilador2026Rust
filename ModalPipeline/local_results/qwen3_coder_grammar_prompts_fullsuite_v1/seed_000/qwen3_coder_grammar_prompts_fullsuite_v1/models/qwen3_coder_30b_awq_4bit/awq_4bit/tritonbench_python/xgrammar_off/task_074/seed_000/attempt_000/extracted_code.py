import torch
import triton
import triton.language as tl

@triton.jit
def _normalize_and_compute_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr,
    x1_size, x2_size, dim_size,
    dim, p_norm, eps_norm, eps_similarity,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting index for this block
    start_idx = block_idx * BLOCK_SIZE
    
    # Load x1 and x2 data
    x1_block = tl.load(x1_ptr + start_idx, mask=start_idx < x1_size)
    x2_block = tl.load(x2_ptr + start_idx, mask=start_idx < x2_size)
    
    # Compute L_p norm
    norm_x1 = tl.sum(tl.abs(x1_block) ** p_norm, axis=0) ** (1.0 / p_norm)
    norm_x2 = tl.sum(tl.abs(x2_block) ** p_norm, axis=0) ** (1.0 / p_norm)
    
    # Normalize vectors
    x1_normalized = x1_block / (norm_x1 + eps_norm)
    x2_normalized = x2_block / (norm_x2 + eps_norm)
    
    # Compute cosine similarity
    dot_product = tl.sum(x1_normalized * x2_normalized, axis=0)
    similarity = dot_product / (norm_x1 * norm_x2 + eps_similarity)
    
    # Store result
    tl.store(output_ptr + block_idx, similarity)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure inputs are on the same device and are contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Get the size of the specified dimension
    dim_size = x1.size(dim)
    
    # Compute the total number of elements
    total_elements = x1.numel()
    
    # Create output tensor
    output = torch.empty((), dtype=torch.float32, device=x1.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    _normalize_and_compute_similarity_kernel[grid](
        x1, x2, output,
        x1.numel(), x2.numel(), dim_size,
        dim, p_norm, eps_norm, eps_similarity,
        BLOCK_SIZE
    )
    
    return output

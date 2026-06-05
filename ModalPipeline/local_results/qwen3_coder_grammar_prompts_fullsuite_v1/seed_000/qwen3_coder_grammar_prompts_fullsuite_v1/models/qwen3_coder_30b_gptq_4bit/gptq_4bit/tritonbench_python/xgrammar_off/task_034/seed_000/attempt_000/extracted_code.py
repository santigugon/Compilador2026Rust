import torch
import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_tanh_kernel(
    input_indices_ptr,
    weight_ptr,
    other_ptr,
    output_ptr,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
    input_indices_size,
    weight_size_0,
    weight_size_1,
    other_size_0,
    other_size_1,
    other_size_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, input_indices_size)
    
    # Load indices
    indices = tl.load(input_indices_ptr + block_start, mask=block_start < block_end)
    
    # Process each index
    for i in range(block_start, block_end):
        idx = indices[i - block_start]
        if idx == padding_idx:
            continue
            
        # Load embedding
        embedding = tl.load(weight_ptr + idx * weight_size_1, mask=idx < weight_size_0)
        
        # Load other tensor (broadcasting)
        other_val = tl.load(other_ptr + (i % other_size_0) * other_size_1 + (i % other_size_1) * other_size_2, mask=True)
        
        # Add other tensor
        result = embedding + other_val
        
        # Apply max_norm if needed
        if max_norm > 0:
            norm = tl.sqrt(tl.sum(result * result))
            scale = tl.where(norm > max_norm, max_norm / (norm + 1e-8), 1.0)
            result = result * scale
            
        # Apply tanh
        result = tl.tanh(result)
        
        # Store result
        tl.store(output_ptr + i * weight_size_1, result)

def fused_embedding_add_tanh(
    input_indices,
    weight,
    other,
    *,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    out=None
):
    # Validate inputs
    if input_indices.dim() == 0:
        input_indices = input_indices.unsqueeze(0)
    
    # Flatten input indices for processing
    input_indices_flat = input_indices.view(-1)
    
    # Get dimensions
    V, D = weight.shape
    other_shape = other.shape
    
    # Determine output shape
    output_shape = list(input_indices.shape) + [D]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=torch.float32, device=weight.device)
    
    # Set up kernel parameters
    BLOCK_SIZE = 1024
    num_blocks = (input_indices_flat.size(0) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_embedding_add_tanh_kernel[
        num_blocks
    ](
        input_indices_flat,
        weight,
        other,
        out,
        padding_idx if padding_idx is not None else -1,
        max_norm if max_norm is not None else 0.0,
        norm_type,
        scale_grad_by_freq,
        sparse,
        input_indices_flat.size(0),
        V,
        D,
        other_shape[0],
        other_shape[1],
        other_shape[2] if len(other_shape) > 2 else 1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

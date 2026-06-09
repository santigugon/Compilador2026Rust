import torch
import triton
import triton.language as tl

@triton.jit
def _fused_embedding_add_tanh_kernel(
    indices_ptr,
    weight_ptr,
    other_ptr,
    out_ptr,
    n_indices: tl.constexpr,
    weight_size_1: tl.constexpr,
    other_shape_0: tl.constexpr,
    other_shape_1: tl.constexpr,
    padding_idx: tl.constexpr,
    max_norm: tl.constexpr,
    norm_type: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_indices
    
    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Check for padding index
    if padding_idx >= 0:
        padding_mask = indices == padding_idx
        mask = mask & ~padding_mask
    
    # Embedding lookup
    # We need to handle the case where indices are out of bounds
    indices = tl.where(indices < 0, indices + weight_size_1, indices)
    indices = tl.clamp(indices, 0, weight_size_1 - 1)
    
    # Create embedding indices
    emb_offsets = indices * weight_size_1
    
    # Load embedding
    emb = tl.load(weight_ptr + emb_offsets, mask=mask, other=0.0)
    
    # Load other tensor (broadcasting)
    other_offsets = (offsets % other_shape_0) * other_shape_1
    other = tl.load(other_ptr + other_offsets, mask=mask, other=0.0)
    
    # Add other tensor
    result = emb + other
    
    # Apply tanh
    result = 2.0 / (1.0 + tl.exp(-2.0 * result)) - 1.0
    
    # Apply max_norm if specified
    if max_norm > 0:
        if norm_type == 2.0:
            norm = tl.sqrt(tl.sum(result * result, axis=0))
        else:
            norm = tl.pow(tl.sum(tl.pow(tl.abs(result), norm_type), axis=0), 1.0 / norm_type)
        scale = max_norm / (norm + 1e-8)
        result = tl.where(norm > max_norm, result * scale, result)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

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
    # Handle scalar padding_idx
    if padding_idx is not None and not isinstance(padding_idx, int):
        padding_idx = int(padding_idx)
    
    # Handle scalar max_norm
    if max_norm is not None and not isinstance(max_norm, (int, float)):
        max_norm = float(max_norm)
    
    # Handle scalar norm_type
    if not isinstance(norm_type, (int, float)):
        norm_type = float(norm_type)
    
    # Determine output shape
    input_shape = input_indices.shape
    weight_shape = weight.shape
    other_shape = other.shape
    
    # Compute output shape
    # The output shape is the same as input_indices shape, but with the last dimension
    # replaced by the embedding dimension
    output_shape = input_shape + (weight_shape[1],)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    else:
        if out.shape != output_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {output_shape}")
    
    # Flatten input indices for processing
    n_indices = input_indices.numel()
    
    # Handle special cases
    if n_indices == 0:
        return out
    
    # Set up kernel launch parameters
    block = 256
    grid = (triton.cdiv(n_indices, block),)
    
    # Launch kernel
    _fused_embedding_add_tanh_kernel[grid](
        input_indices,
        weight,
        other,
        out,
        n_indices,
        weight_shape[1],
        other_shape[0],
        other_shape[1],
        padding_idx if padding_idx is not None else -1,
        max_norm if max_norm is not None else 0.0,
        norm_type,
        BLOCK=block
    )
    
    return out

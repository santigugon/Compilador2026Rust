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
    embedding_dim: tl.constexpr,
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
    else:
        padding_mask = tl.zeros(BLOCK, dtype=tl.bool)
    
    # Embedding lookup
    # Create a block of indices for embedding lookup
    indices_expanded = indices[:, None]  # [BLOCK, 1]
    
    # Load embedding vectors
    weight_offsets = indices_expanded * embedding_dim + tl.arange(0, embedding_dim)[None, :]
    weight_mask = (indices >= 0) & (indices < embedding_dim)  # Simple check for valid indices
    
    # Load weight values
    weight_values = tl.load(weight_ptr + weight_offsets, mask=weight_mask[:, None], other=0.0)
    
    # Load other tensor (broadcasting)
    other_values = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise addition
    result = weight_values + other_values[:, None]
    
    # Apply tanh
    result = 2.0 / (1.0 + tl.exp(-2.0 * result)) - 1.0
    
    # Apply max_norm if specified
    if max_norm > 0:
        norms = tl.sqrt(tl.sum(result * result, axis=1, keepdims=True))
        scale = tl.where(norms > max_norm, max_norm / (norms + 1e-8), 1.0)
        result = result * scale
    
    # Store result
    out_offsets = offsets[:, None] * embedding_dim + tl.arange(0, embedding_dim)[None, :]
    tl.store(out_ptr + out_offsets, result, mask=mask[:, None])

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
    
    # Validate inputs
    if input_indices.dim() == 0:
        input_indices = input_indices.unsqueeze(0)
    
    # Determine output shape
    input_shape = input_indices.shape
    embedding_dim = weight.shape[1]
    output_shape = input_shape + (embedding_dim,)
    
    # Create output tensor
    if out is not None:
        if out.shape != output_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {output_shape}")
        out = out
    else:
        out = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Flatten indices for processing
    n_indices = input_indices.numel()
    
    # Handle special case where there are no indices
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
        embedding_dim,
        padding_idx if padding_idx is not None else -1,
        max_norm if max_norm is not None else 0.0,
        norm_type,
        BLOCK=block
    )
    
    return out

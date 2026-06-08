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
        indices = tl.where(padding_mask, 0, indices)
    
    # Compute embedding lookup
    # Each index maps to a row in weight matrix
    embedding_offsets = indices * embedding_dim
    embeddings = tl.zeros((BLOCK, embedding_dim), dtype=tl.float32)
    
    # Load embeddings with proper masking
    for i in range(embedding_dim):
        # Load weight values for each embedding dimension
        weight_offsets = embedding_offsets + i
        weight_values = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
        embeddings = tl.where(mask, embeddings + weight_values, embeddings)
    
    # Load other tensor (broadcasting)
    other_values = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Add other tensor to embeddings
    result = embeddings + other_values
    
    # Apply tanh activation
    result = 2.0 / (1.0 + tl.exp(-2.0 * result)) - 1.0
    
    # Apply max_norm if specified
    if max_norm > 0:
        # Compute norm
        norm = tl.sqrt(tl.sum(result * result, axis=1))
        # Normalize if norm exceeds max_norm
        scale = max_norm / (norm + 1e-8)
        scale = tl.where(norm > max_norm, scale, 1.0)
        result = result * scale[:, None]
    
    # Store result
    out_offsets = offsets * embedding_dim
    for i in range(embedding_dim):
        out_offsets_i = out_offsets + i
        tl.store(out_ptr + out_offsets_i, result[:, i], mask=mask)

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
    
    # Compute output shape
    output_shape = list(input_indices.shape)
    output_shape.append(weight.shape[1])
    
    # Create output tensor
    if out is not None:
        out = torch.empty_like(out)
    else:
        out = torch.empty(output_shape, dtype=torch.float32, device=weight.device)
    
    # Get dimensions
    n_indices = input_indices.numel()
    embedding_dim = weight.shape[1]
    
    # Handle special cases
    if n_indices == 0:
        return out
    
    # Set up kernel launch parameters
    BLOCK = 256
    grid = (triton.cdiv(n_indices, BLOCK),)
    
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
        BLOCK=BLOCK
    )
    
    return out

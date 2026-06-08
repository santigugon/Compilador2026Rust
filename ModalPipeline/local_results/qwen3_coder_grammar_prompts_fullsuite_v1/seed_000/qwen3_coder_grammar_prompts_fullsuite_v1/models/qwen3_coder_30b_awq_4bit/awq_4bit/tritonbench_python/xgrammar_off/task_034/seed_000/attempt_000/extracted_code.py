import torch
import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_tanh_kernel(
    indices_ptr,
    weight_ptr,
    other_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    num_indices,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices
    
    indices = tl.load(indices_ptr + offsets, mask=mask)
    
    # Handle padding index
    if padding_idx is not None:
        padding_mask = indices == padding_idx
        indices = tl.where(padding_mask, 0, indices)
    
    # Embedding lookup
    embedding_offsets = indices * embedding_dim
    embeddings = tl.load(weight_ptr + embedding_offsets[:, None] + tl.arange(0, embedding_dim), mask=mask[:, None])
    
    # Add other tensor
    other = tl.load(other_ptr + tl.arange(0, embedding_dim), mask=tl.arange(0, embedding_dim) < embedding_dim)
    result = embeddings + other[None, :]
    
    # Apply tanh
    result = tl.tanh(result)
    
    # Apply max_norm if specified
    if max_norm is not None:
        norms = tl.sqrt(tl.sum(result * result, axis=1, keepdims=True))
        scale = max_norm / (norms + 1e-8)
        scale = tl.minimum(scale, 1.0)
        result = result * scale
    
    # Store result
    tl.store(output_ptr + offsets[:, None] + tl.arange(0, embedding_dim), result, mask=mask[:, None])

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
    
    # Flatten indices for processing
    flat_indices = input_indices.view(-1)
    num_indices = flat_indices.shape[0]
    embedding_dim = weight.shape[1]
    num_embeddings = weight.shape[0]
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(flat_indices.shape[0], embedding_dim, dtype=torch.float32, device=weight.device)
    else:
        out = out.view(-1, embedding_dim)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_blocks = (num_indices + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_add_tanh_kernel[
        num_blocks,
        1,
        (weight.device.index if weight.device.type == 'cuda' else 0),
    ](
        flat_indices,
        weight,
        other,
        out,
        num_embeddings,
        embedding_dim,
        num_indices,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
        BLOCK_SIZE
    )
    
    return out.view(input_indices.shape + (embedding_dim,))

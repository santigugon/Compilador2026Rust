import torch
import triton
import triton.language as tl

def _get_numel(x):
    return x.numel()

def _get_stride(x, dim):
    return x.stride(dim) if x.dim() > dim else 0

@triton.jit
def _fused_embedding_add_tanh_kernel(
    indices_ptr,
    weight_ptr,
    other_ptr,
    out_ptr,
    indices_n,
    weight_v,  # V
    weight_d,  # D
    other_shape,
    other_strides,
    padding_idx,
    max_norm,
    norm_type,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < indices_n
    
    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Handle padding index
    if padding_idx >= 0:
        padding_mask = indices == padding_idx
        
    # Compute embedding indices
    emb_indices = indices
    
    # Load embeddings
    emb_offsets = emb_indices * weight_d
    embeddings = tl.load(weight_ptr + emb_offsets, mask=mask, other=0.0)
    
    # Load other tensor (broadcasting)
    # For simplicity, we assume other is broadcastable to (indices_n, D)
    # In practice, this would require more complex broadcasting logic
    other_offsets = offsets * other_strides[0] if len(other_strides) > 0 else 0
    other_vals = tl.load(other_ptr + other_offsets, mask=mask, other=0.0)
    
    # Add other tensor
    result = embeddings + other_vals
    
    # Apply tanh
    result = 2.0 / (1.0 + tl.exp(-2.0 * result)) - 1.0
    
    # Apply max_norm if specified
    if max_norm > 0.0:
        norm = tl.sqrt(tl.sum(result * result, axis=0))
        norm = tl.maximum(norm, 1e-12)
        scale = max_norm / norm
        result = result * scale
    
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
    # Validate inputs
    if input_indices.dim() == 0:
        input_indices = input_indices.unsqueeze(0)
    
    # Get dimensions
    indices_n = _get_numel(input_indices)
    weight_v, weight_d = weight.shape
    
    # Handle output tensor
    if out is None:
        out = torch.empty(input_indices.shape + (weight_d,), dtype=weight.dtype, device=weight.device)
    else:
        assert out.shape == input_indices.shape + (weight_d,), "Output shape mismatch"
        assert out.dtype == weight.dtype, "Output dtype mismatch"
        assert out.device == weight.device, "Output device mismatch"
    
    # Handle padding_idx
    if padding_idx is None:
        padding_idx = -1
    else:
        padding_idx = int(padding_idx)
    
    # Handle max_norm
    if max_norm is None:
        max_norm = 0.0
    else:
        max_norm = float(max_norm)
    
    # Handle other tensor
    # Broadcast other tensor to match embedding shape
    # This is a simplified version - in practice, more complex broadcasting logic would be needed
    other_shape = other.shape
    other_strides = [other.stride(i) for i in range(other.dim())]
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(indices_n, block),)
    
    _fused_embedding_add_tanh_kernel[grid](
        input_indices,
        weight,
        other,
        out,
        indices_n,
        weight_v,
        weight_d,
        other_shape,
        other_strides,
        padding_idx,
        max_norm,
        norm_type,
        BLOCK=block
    )
    
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
#     embeddings = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
#     sum_embeddings = embeddings + other
#     result = torch.tanh(sum_embeddings)
#     if out is not None:
#         out.copy_(result)
#     return result

def test_fused_embedding_add_tanh():
    results = {}

    # Test case 1: Basic test without padding_idx, max_norm, scale_grad_by_freq, sparse, and out
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_1"] = fused_embedding_add_tanh(input_indices, weight, other)

    # Test case 2: Test with padding_idx
    padding_idx = 0
    input_indices = torch.tensor([0, 1, 2], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_2"] = fused_embedding_add_tanh(input_indices, weight, other, padding_idx=padding_idx)

    # Test case 3: Test with max_norm
    max_norm = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_3"] = fused_embedding_add_tanh(input_indices, weight, other, max_norm=max_norm)

    # Test case 4: Test with norm_type
    norm_type = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_4"] = fused_embedding_add_tanh(input_indices, weight, other, norm_type=norm_type)

    return results

test_results = test_fused_embedding_add_tanh()

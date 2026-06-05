import torch
import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_tanh_kernel(
    indices_ptr, weight_ptr, other_ptr, output_ptr,
    indices_size, weight_size, other_size,
    padding_idx, max_norm, norm_type,
    scale_grad_by_freq, sparse,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < indices_size
    
    indices = tl.load(indices_ptr + offsets, mask=mask)
    
    # Handle padding index
    if padding_idx >= 0:
        padding_mask = indices == padding_idx
        indices = tl.where(padding_mask, 0, indices)
    
    # Embedding lookup
    embedding_indices = indices
    weight_ptr += embedding_indices[:, None] * weight_size[1]
    
    # Load embedding
    embedding = tl.load(weight_ptr, mask=mask[:, None])
    
    # Add other tensor (broadcasting)
    other = tl.load(other_ptr, mask=mask[:, None])
    result = embedding + other
    
    # Apply tanh
    result = tl.tanh(result)
    
    # Apply max_norm if specified
    if max_norm > 0:
        norms = tl.sqrt(tl.sum(result * result, axis=1, keep_dims=True))
        scale = max_norm / (norms + 1e-8)
        scale = tl.where(norms > max_norm, scale, 1.0)
        result = result * scale
    
    # Store result
    tl.store(output_ptr + offsets[:, None], result, mask=mask[:, None])

def fused_embedding_add_tanh(
    input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None
):
    if out is None:
        out = torch.empty(input_indices.shape + (weight.shape[1],), dtype=torch.float32, device=input_indices.device)
    
    if max_norm is None:
        max_norm = 0.0
    
    if padding_idx is None:
        padding_idx = -1
    
    # Ensure inputs are contiguous
    input_indices = input_indices.contiguous()
    weight = weight.contiguous()
    other = other.contiguous()
    out = out.contiguous()
    
    # Launch kernel
    grid = (triton.cdiv(input_indices.numel(), 1024),)
    fused_embedding_add_tanh_kernel[grid](
        input_indices, weight, other, out,
        input_indices.numel(), weight.shape, other.shape,
        padding_idx, max_norm, norm_type,
        scale_grad_by_freq, sparse,
        BLOCK_SIZE=1024
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

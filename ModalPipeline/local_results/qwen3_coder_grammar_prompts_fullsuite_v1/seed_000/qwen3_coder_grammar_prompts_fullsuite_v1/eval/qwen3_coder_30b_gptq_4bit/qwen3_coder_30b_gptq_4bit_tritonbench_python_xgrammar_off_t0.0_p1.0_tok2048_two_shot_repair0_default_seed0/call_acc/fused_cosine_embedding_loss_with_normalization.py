import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute L2 norm
    squared = x * x
    sum_squared = tl.sum(squared, axis=0)
    norm = tl.sqrt(sum_squared + 1e-8)
    # Normalize
    normalized = x / norm
    tl.store(out_ptr + offsets, normalized, mask=mask)

@triton.jit
def _cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, out_ptr, 
    n: tl.constexpr, margin: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x1 = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(input2_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine similarity
    dot_product = x1 * x2
    sum_dot = tl.sum(dot_product, axis=0)
    
    # Compute norms
    x1_squared = x1 * x1
    x2_squared = x2 * x2
    sum_x1_squared = tl.sum(x1_squared, axis=0)
    sum_x2_squared = tl.sum(x2_squared, axis=0)
    
    norm1 = tl.sqrt(sum_x1_squared + 1e-8)
    norm2 = tl.sqrt(sum_x2_squared + 1e-8)
    
    # Compute cosine similarity
    cosine_sim = sum_dot / (norm1 * norm2 + 1e-8)
    
    # Compute loss
    loss = tl.where(target == 1, 
                    1.0 - cosine_sim,
                    tl.maximum(0, cosine_sim - margin))
    
    tl.store(out_ptr + offsets, loss, mask=mask)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor, 
    input2: torch.Tensor, 
    target: torch.Tensor, 
    margin: float = 0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    # Normalize inputs
    input1_norm = torch.empty_like(input1)
    input2_norm = torch.empty_like(input2)
    
    n = input1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Normalize both inputs
    _l2_norm_kernel[grid](input1, input1_norm, n, BLOCK=block)
    _l2_norm_kernel[grid](input2, input2_norm, n, BLOCK=block)
    
    # Compute cosine embedding loss
    out = torch.empty_like(input1_norm)
    _cosine_embedding_loss_kernel[grid](
        input1_norm, input2_norm, target, out, n, margin, BLOCK=block
    )
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(out)
    elif reduction == 'sum':
        return torch.sum(out)
    else:  # reduction == 'none'
        return out

##################################################################################################################################################



import torch
import torch.nn.functional as F
import torch

def test_fused_cosine_embedding_loss_with_normalization():
    results = {}

    # Test case 1: Default margin and reduction
    input1 = torch.randn(3, 5, device='cuda', requires_grad=True)
    input2 = torch.randn(3, 5, device='cuda', requires_grad=True)
    target = torch.tensor([1, -1, 1], device='cuda')
    results["test_case_1"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target)

    # Test case 2: Margin > 0
    margin = 0.5
    results["test_case_2"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin=margin)

    # Test case 3: Reduction 'sum'
    reduction = 'sum'
    results["test_case_3"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, reduction=reduction)

    # Test case 4: Reduction 'none'
    reduction = 'none'
    results["test_case_4"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, reduction=reduction)

    return results

test_results = test_fused_cosine_embedding_loss_with_normalization()

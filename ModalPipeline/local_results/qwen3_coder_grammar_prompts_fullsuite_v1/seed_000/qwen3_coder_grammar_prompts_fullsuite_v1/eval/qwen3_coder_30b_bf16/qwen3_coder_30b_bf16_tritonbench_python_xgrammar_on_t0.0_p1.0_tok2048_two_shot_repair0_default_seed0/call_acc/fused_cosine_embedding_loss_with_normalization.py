import torch
import triton
import triton.language as tl

def _l2_normalize_kernel(x_ptr, out_ptr, n: tl.constexpr, dim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute L2 norm
    x_sq = x * x
    sum_sq = tl.sum(x_sq)
    norm = tl.sqrt(sum_sq)
    # Avoid division by zero
    norm = tl.where(norm == 0, 1.0, norm)
    normalized = x / norm
    tl.store(out_ptr + offsets, normalized, mask=mask)

def _cosine_embedding_loss_kernel(x1_ptr, x2_ptr, target_ptr, out_ptr, n: tl.constexpr, margin: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine similarity
    dot_product = x1 * x2
    cos_sim = tl.sum(dot_product)
    
    # Compute loss
    loss = tl.where(target > 0, 1.0 - cos_sim, tl.maximum(0.0, cos_sim - margin))
    tl.store(out_ptr + offsets, loss, mask=mask)

@triton.jit
def _sum_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use atomic add for reduction
    tl.atomic_add(out_ptr, tl.sum(x))

@triton.jit
def _mean_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use atomic add for reduction
    tl.atomic_add(out_ptr, tl.sum(x))
    # Store the mean
    tl.store(out_ptr + 1, n)

@triton.jit
def _none_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    # Normalize inputs
    input1_norm = torch.empty_like(input1)
    input2_norm = torch.empty_like(input2)
    
    n = input1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Normalize both inputs
    _l2_normalize_kernel[grid](input1, input1_norm, n, 0, BLOCK=block)
    _l2_normalize_kernel[grid](input2, input2_norm, n, 0, BLOCK=block)
    
    # Compute cosine embedding loss
    loss = torch.empty_like(input1_norm)
    _cosine_embedding_loss_kernel[grid](input1_norm, input2_norm, target, loss, n, margin, BLOCK=block)
    
    # Apply reduction
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'mean':
        return torch.mean(loss)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
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

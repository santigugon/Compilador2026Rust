import torch
import triton
import triton.language as tl

@triton.jit
def _l2_normalize_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute L2 norm
    norm = tl.sqrt(tl.sum(x * x, axis=0) + 1e-8)
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
    
    input1 = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    input2 = tl.load(input2_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine similarity
    dot_product = tl.sum(input1 * input2, axis=0)
    norm1 = tl.sqrt(tl.sum(input1 * input1, axis=0) + 1e-8)
    norm2 = tl.sqrt(tl.sum(input2 * input2, axis=0) + 1e-8)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
    
    # Compute loss
    loss = tl.where(
        target == 1,
        1 - cosine_sim,
        tl.maximum(cosine_sim - margin, 0)
    )
    
    tl.store(out_ptr + offsets, loss, mask=mask)

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    # Normalize inputs
    input1_norm = torch.empty_like(input1)
    input2_norm = torch.empty_like(input2)
    
    n = input1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Normalize both inputs
    _l2_normalize_kernel[grid](input1, input1_norm, n, BLOCK=block)
    _l2_normalize_kernel[grid](input2, input2_norm, n, BLOCK=block)
    
    # Compute cosine embedding loss
    loss = torch.empty_like(input1_norm)
    _cosine_embedding_loss_kernel[grid](input1_norm, input2_norm, target, loss, n, margin, BLOCK=block)
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:  # reduction == 'none'
        return loss

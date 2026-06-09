import torch
import triton
import triton.language as tl

def _l2_normalize_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
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
    sum_dot = tl.sum(dot_product)
    
    # Compute cosine similarity
    x1_sq = x1 * x1
    x2_sq = x2 * x2
    sum_x1_sq = tl.sum(x1_sq)
    sum_x2_sq = tl.sum(x2_sq)
    norm_x1 = tl.sqrt(sum_x1_sq)
    norm_x2 = tl.sqrt(sum_x2_sq)
    
    # Avoid division by zero
    norm_x1 = tl.where(norm_x1 == 0, 1.0, norm_x1)
    norm_x2 = tl.where(norm_x2 == 0, 1.0, norm_x2)
    
    cosine_sim = sum_dot / (norm_x1 * norm_x2)
    
    # Compute loss
    loss = tl.where(target == 1.0, 1.0 - cosine_sim, 
                    tl.maximum(0.0, cosine_sim - margin))
    
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
    
    # Store the sum in the first element
    if pid == 0:
        tl.store(out_ptr, tl.sum(x))

@triton.jit
def _mean_kernel_final(out_ptr, sum_ptr, n: tl.constexpr):
    if tl.program_id(0) == 0:
        tl.store(out_ptr, tl.load(sum_ptr) / n)

@triton.jit
def _sum_kernel_final(out_ptr, sum_ptr):
    if tl.program_id(0) == 0:
        tl.store(out_ptr, tl.load(sum_ptr))

@triton.jit
def _none_kernel(x1_ptr, x2_ptr, target_ptr, out_ptr, n: tl.constexpr, margin: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine similarity
    dot_product = x1 * x2
    sum_dot = tl.sum(dot_product)
    
    # Compute cosine similarity
    x1_sq = x1 * x1
    x2_sq = x2 * x2
    sum_x1_sq = tl.sum(x1_sq)
    sum_x2_sq = tl.sum(x2_sq)
    norm_x1 = tl.sqrt(sum_x1_sq)
    norm_x2 = tl.sqrt(sum_x2_sq)
    
    # Avoid division by zero
    norm_x1 = tl.where(norm_x1 == 0, 1.0, norm_x1)
    norm_x2 = tl.where(norm_x2 == 0, 1.0, norm_x2)
    
    cosine_sim = sum_dot / (norm_x1 * norm_x2)
    
    # Compute loss
    loss = tl.where(target == 1.0, 1.0 - cosine_sim, 
                    tl.maximum(0.0, cosine_sim - margin))
    
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
    out = torch.empty_like(input1)
    _cosine_embedding_loss_kernel[grid](input1_norm, input2_norm, target, out, n, margin, BLOCK=block)
    
    # Apply reduction
    if reduction == 'none':
        return out
    elif reduction == 'sum':
        return torch.sum(out)
    elif reduction == 'mean':
        return torch.mean(out)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
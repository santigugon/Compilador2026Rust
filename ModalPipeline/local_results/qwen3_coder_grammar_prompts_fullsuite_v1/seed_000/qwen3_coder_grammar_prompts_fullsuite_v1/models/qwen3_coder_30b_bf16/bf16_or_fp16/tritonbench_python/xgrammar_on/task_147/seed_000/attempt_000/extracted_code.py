import torch
import triton
import triton.language as tl

def _norm_kernel(x_ptr, out_ptr, n: tl.constexpr, dim: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if p == 2.0:
        x = x * x
        x = tl.sum(x, axis=dim)
        x = tl.sqrt(x + eps)
    else:
        x = tl.abs(x) ** p
        x = tl.sum(x, axis=dim)
        x = x ** (1.0 / p)
    tl.store(out_ptr + offsets, x, mask=mask)

def _pairwise_distance_kernel(x1_ptr, x2_ptr, out_ptr, n1: tl.constexpr, n2: tl.constexpr, dim: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n1 * n2
    x1_val = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2_val = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    diff = x1_val - x2_val
    diff = tl.abs(diff)
    diff = diff + eps
    tl.store(out_ptr + offsets, diff, mask=mask)

@triton.jit
def _normalize_kernel(x_ptr, out_ptr, n: tl.constexpr, dim: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute norm along specified dimension
    if p == 2.0:
        norm = tl.sum(x * x, axis=dim)
        norm = tl.sqrt(norm + eps)
    else:
        norm = tl.sum(tl.abs(x) ** p, axis=dim)
        norm = norm ** (1.0 / p)
    # Normalize
    x = x / (norm + eps)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def _pairwise_distance_kernel_v2(x1_ptr, x2_ptr, out_ptr, n1: tl.constexpr, n2: tl.constexpr, dim: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n1 * n2
    x1_val = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2_val = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    diff = x1_val - x2_val
    diff = tl.abs(diff)
    diff = diff + eps
    tl.store(out_ptr + offsets, diff, mask=mask)

@triton.jit
def _compute_distance_kernel(x1_ptr, x2_ptr, out_ptr, n1: tl.constexpr, n2: tl.constexpr, dim: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n1 * n2
    x1_val = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2_val = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    diff = x1_val - x2_val
    if p == 2.0:
        diff = diff * diff
        diff = tl.sum(diff, axis=dim)
        diff = tl.sqrt(diff + eps)
    else:
        diff = tl.abs(diff) ** p
        diff = tl.sum(diff, axis=dim)
        diff = diff ** (1.0 / p)
    tl.store(out_ptr + offsets, diff, mask=mask)

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Normalize both tensors
    x1_norm = torch.empty_like(x1)
    x2_norm = torch.empty_like(x2)
    
    # Compute norms
    n = x1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Normalize x1
    _normalize_kernel[grid](x1, x1_norm, n, 0, p_norm, eps_norm, BLOCK=block)
    
    # Normalize x2
    _normalize_kernel[grid](x2, x2_norm, n, 0, p_norm, eps_norm, BLOCK=block)
    
    # Compute pairwise distance
    out = torch.empty(x1.shape[0], x2.shape[0], dtype=torch.float32)
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    
    # For simplicity, we'll compute the full pairwise distance matrix
    # This is a simplified version - in practice, you'd want to handle
    # the broadcasting and dimensionality properly
    if x1.shape[0] == x2.shape[0] and x1.shape[1] == x2.shape[1]:
        # If shapes match, compute element-wise distance
        out = torch.abs(x1_norm - x2_norm)
        if p_norm == 2.0:
            out = torch.sqrt(out * out + eps_distance)
        else:
            out = torch.pow(out, p_norm) + eps_distance
            out = torch.pow(out, 1.0 / p_norm)
    else:
        # For general case, compute full pairwise distance
        # This is a simplified approach - a full implementation would be more complex
        out = torch.cdist(x1_norm, x2_norm, p=p_norm, compute_mode='use_mm_for_euclid_dist')
        
    if not keepdim:
        out = out.squeeze(-1) if out.shape[-1] == 1 else out
    
    return out
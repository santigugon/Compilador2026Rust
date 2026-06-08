import torch
import triton
import triton.language as tl

@triton.jit
def _normalize_kernel(x_ptr, out_ptr, n, dim_size, p_norm, eps_norm, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute norm along the specified dimension
    if p_norm == 2.0:
        norm = tl.sqrt(tl.sum(x * x) + eps_norm)
    elif p_norm == 1.0:
        norm = tl.sum(tl.abs(x)) + eps_norm
    else:
        norm = tl.pow(tl.sum(tl.pow(tl.abs(x), p_norm)) + eps_norm, 1.0 / p_norm)
    
    # Normalize
    normalized = x / norm
    tl.store(out_ptr + offsets, normalized, mask=mask)

@triton.jit
def _pairwise_distance_kernel(x1_ptr, x2_ptr, out_ptr, n1, n2, dim_size, p_norm, eps_distance, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n1 * n2
    
    # Compute pairwise distances
    if mask:
        i = offsets // n2
        j = offsets % n2
        diff = x1_ptr[i] - x2_ptr[j]
        if p_norm == 2.0:
            dist = tl.sqrt(tl.sum(diff * diff) + eps_distance)
        elif p_norm == 1.0:
            dist = tl.sum(tl.abs(diff)) + eps_distance
        else:
            dist = tl.pow(tl.sum(tl.pow(tl.abs(diff), p_norm)) + eps_distance, 1.0 / p_norm)
        tl.store(out_ptr + offsets, dist)

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Normalize both tensors
    x1_normalized = torch.empty_like(x1)
    x2_normalized = torch.empty_like(x2)
    
    # Flatten tensors for normalization
    x1_flat = x1.view(-1, x1.size(-1))
    x2_flat = x2.view(-1, x2.size(-1))
    
    n1 = x1_flat.numel()
    n2 = x2_flat.numel()
    dim_size = x1_flat.size(-1)
    
    # Normalize x1
    block = 256
    grid1 = (triton.cdiv(n1, block),)
    _normalize_kernel[grid1](x1_flat, x1_normalized.view(-1, dim_size), n1, dim_size, p_norm, eps_norm, BLOCK=block)
    
    # Normalize x2
    grid2 = (triton.cdiv(n2, block),)
    _normalize_kernel[grid2](x2_flat, x2_normalized.view(-1, dim_size), n2, dim_size, p_norm, eps_norm, BLOCK=block)
    
    # Compute pairwise distances
    out = torch.empty(x1.size(0), x2.size(0), dtype=torch.float32, device=x1.device)
    
    # Reshape for pairwise computation
    x1_expanded = x1_normalized.unsqueeze(1)  # [batch1, 1, features]
    x2_expanded = x2_normalized.unsqueeze(0)  # [1, batch2, features]
    
    # Compute distances using broadcasting
    diff = x1_expanded - x2_expanded  # [batch1, batch2, features]
    
    if p_norm == 2.0:
        distances = torch.sqrt(torch.sum(diff * diff, dim=-1) + eps_distance)
    elif p_norm == 1.0:
        distances = torch.sum(torch.abs(diff), dim=-1) + eps_distance
    else:
        distances = torch.pow(torch.sum(torch.pow(torch.abs(diff), p_norm), dim=-1) + eps_distance, 1.0 / p_norm)
    
    return distances

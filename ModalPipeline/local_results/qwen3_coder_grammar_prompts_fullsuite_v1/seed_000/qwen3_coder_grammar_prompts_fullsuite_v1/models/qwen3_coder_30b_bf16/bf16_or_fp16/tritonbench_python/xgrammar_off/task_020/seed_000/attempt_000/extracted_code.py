import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(x1_ptr, x2_ptr, out_ptr, n1: tl.constexpr, n2: tl.constexpr, d: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute distances for one pair of vectors
    if pid < n1 and pid2 < n2:
        # Load x1 vector
        x1_offsets = pid * d + tl.arange(0, BLOCK)
        x1_mask = x1_offsets < n1 * d
        x1 = tl.load(x1_ptr + x1_offsets, mask=x1_mask, other=0.0)
        
        # Load x2 vector
        x2_offsets = pid2 * d + tl.arange(0, BLOCK)
        x2_mask = x2_offsets < n2 * d
        x2 = tl.load(x2_ptr + x2_offsets, mask=x2_mask, other=0.0)
        
        # Compute distance
        diff = x1 - x2
        if p == 2.0:
            dist = tl.sum(diff * diff)
        else:
            dist = tl.sum(tl.abs(diff) ** p)
        
        dist = tl.sqrt(dist) + eps
        tl.store(out_ptr + pid * n2 + pid2, dist)

@triton.jit
def _normalize_kernel(x_ptr, out_ptr, n1: tl.constexpr, n2: tl.constexpr, d: tl.constexpr, dim_norm: tl.constexpr, p_norm: tl.constexpr, eps_norm: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Normalize along the specified dimension
    if pid < n1:
        # Load the row
        offsets = pid * n2 + tl.arange(0, BLOCK)
        mask = offsets < n1 * n2
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute norm along the specified dimension
        if p_norm == 2.0:
            norm = tl.sqrt(tl.sum(x * x))
        else:
            norm = tl.sum(tl.abs(x) ** p_norm)
        
        norm = norm + eps_norm
        
        # Normalize
        normalized = x / norm
        tl.store(out_ptr + offsets, normalized, mask=mask)

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    # Validate inputs
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Get dimensions
    shape = x1.shape
    n1 = shape[0] if len(shape) > 0 else 1
    n2 = shape[1] if len(shape) > 1 else 1
    d = shape[-1] if len(shape) > 0 else 1
    
    # Compute pairwise distances
    out = torch.empty(n1, n2, device=x1.device, dtype=torch.float32)
    
    # For small tensors, use PyTorch directly
    if n1 * n2 <= 1024:
        # Use PyTorch for small tensors
        x1_flat = x1.view(-1, d)
        x2_flat = x2.view(-1, d)
        distances = torch.cdist(x1_flat, x2_flat, p=p_distance, eps=eps_distance)
        out = distances
    else:
        # Use Triton for larger tensors
        block = 256
        grid = (triton.cdiv(n1, block), triton.cdiv(n2, block))
        _pairwise_distance_kernel[grid](x1, x2, out, n1, n2, d, p_distance, eps_distance, BLOCK=block)
    
    # Normalize along specified dimension
    if dim_norm == 1:
        # Normalize along the second dimension (n2)
        out = out.view(n1, n2)
        out_normalized = torch.empty_like(out)
        block = 256
        grid = (triton.cdiv(n1, block),)
        _normalize_kernel[grid](out, out_normalized, n1, n2, n2, dim_norm, p_norm, eps_norm, BLOCK=block)
        out = out_normalized
    else:
        # For other dimensions, use PyTorch
        out = torch.nn.functional.normalize(out, p=p_norm, dim=dim_norm, eps=eps_norm)
    
    # Handle keepdim
    if keepdim:
        out = out.unsqueeze(dim_norm)
    
    return out

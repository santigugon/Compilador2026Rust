import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_norm_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_new = y + alpha * x
    tl.store(y_ptr + offsets, y_new, mask=mask)
    # Compute squared sum for norm
    y_sq = y_new * y_new
    tl.store(out_ptr + offsets, y_sq, mask=mask)

def scaled_add_norm(y, x, alpha):
    # Create output tensor for squared values
    out = torch.empty_like(y)
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel to compute scaled addition and squared values
    _scaled_add_norm_kernel[grid](y, x, out, n, alpha, BLOCK=block)
    
    # Sum all squared values and take square root
    squared_sum = out.sum()
    norm = torch.sqrt(squared_sum)
    
    return norm

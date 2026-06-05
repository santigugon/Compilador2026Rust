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
    out = torch.empty_like(y)
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create a temporary tensor to store squared values
    temp = torch.empty(n, dtype=torch.float32, device=y.device)
    
    _scaled_add_norm_kernel[grid](y, x, temp, n, alpha, BLOCK=block)
    
    # Compute the 2-norm
    squared_sum = temp.sum()
    norm = torch.sqrt(squared_sum)
    
    return norm

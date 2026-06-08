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
    y = y + alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)
    # Compute squared sum for norm
    y_squared = y * y
    tl.store(out_ptr + offsets, y_squared, mask=mask)

def scaled_add_norm(y, x, alpha):
    # Ensure inputs are contiguous and have the same shape
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    
    # Create output tensor for squared values
    out = torch.empty_like(y)
    
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel to perform y += alpha * x and store squared values
    _scaled_add_norm_kernel[grid](y, x, out, n, alpha, BLOCK=block)
    
    # Sum the squared values and compute the 2-norm
    squared_sum = out.sum()
    norm = torch.sqrt(squared_sum)
    
    return norm

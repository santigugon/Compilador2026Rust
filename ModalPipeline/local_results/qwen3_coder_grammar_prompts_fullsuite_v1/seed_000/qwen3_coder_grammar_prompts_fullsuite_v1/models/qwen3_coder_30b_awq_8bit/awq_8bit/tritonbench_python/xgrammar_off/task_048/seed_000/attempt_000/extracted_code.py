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
    
    # Compute y += alpha * x
    y = y + alpha * x
    
    # Store the updated y
    tl.store(y_ptr + offsets, y, mask=mask)
    
    # Compute squared values for norm calculation
    y_squared = y * y
    tl.store(out_ptr + offsets, y_squared, mask=mask)

def scaled_add_norm(y, x, alpha):
    # Ensure tensors are contiguous and have the same shape
    y = y.contiguous()
    x = x.contiguous()
    
    # Create output tensor for squared values
    squared_y = torch.empty_like(y)
    
    # Compute the scaled addition and squared values
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _scaled_add_norm_kernel[grid](y, x, squared_y, n, alpha, BLOCK=block)
    
    # Compute 2-norm of the updated y
    norm = torch.sqrt(torch.sum(squared_y))
    
    return norm

import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_norm_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load y and x
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute y += alpha * x
    y = y + alpha * x
    
    # Store the updated y
    tl.store(y_ptr + offsets, y, mask=mask)
    
    # Compute squared values for norm
    y_squared = y * y
    tl.store(out_ptr + offsets, y_squared, mask=mask)

@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Use atomic add for reduction
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))


def scaled_add_norm(y, x, alpha):
    # Ensure inputs are contiguous and of the same shape
    y = y.contiguous()
    x = x.contiguous()
    
    # Check shapes
    assert y.shape == x.shape, "y and x must have the same shape"
    
    # Create output tensor for squared values
    y_squared = torch.empty_like(y)
    
    # Compute the scaled addition and squared values
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _scaled_add_norm_kernel[grid](y, x, y_squared, n, alpha, BLOCK=block)
    
    # Compute sum of squares
    sum_squared = torch.zeros(1, dtype=torch.float32, device=y.device)
    _reduce_sum_kernel[grid](y_squared, sum_squared, n, BLOCK=block)
    
    # Compute 2-norm
    norm = torch.sqrt(sum_squared)
    
    return norm
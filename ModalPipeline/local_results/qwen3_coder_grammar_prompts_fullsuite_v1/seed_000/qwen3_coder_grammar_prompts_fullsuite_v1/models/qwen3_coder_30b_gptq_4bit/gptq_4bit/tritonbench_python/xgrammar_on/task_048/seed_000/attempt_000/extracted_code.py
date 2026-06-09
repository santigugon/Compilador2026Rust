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

@triton.jit
def _norm_kernel(y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_squared = y * y
    tl.store(out_ptr + offsets, y_squared, mask=mask)

@triton.jit
def _sum_kernel(y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def scaled_add_norm(y, x, alpha):
    # Ensure inputs are contiguous
    y = y.contiguous()
    x = x.contiguous()
    
    # Compute y += alpha * x
    out = torch.empty_like(y)
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # First kernel: compute y += alpha * x
    _scaled_add_norm_kernel[grid](y, x, out, n, alpha, BLOCK=block)
    
    # Second kernel: compute squared values
    squared_out = torch.empty_like(out)
    _norm_kernel[grid](out, squared_out, n, BLOCK=block)
    
    # Third kernel: sum all squared values
    sum_out = torch.empty(1, dtype=torch.float32)
    _sum_kernel[grid](squared_out, sum_out, n, BLOCK=block)
    
    # Reduce sum to scalar
    total_sum = sum_out.sum()
    
    # Compute 2-norm
    norm = torch.sqrt(total_sum)
    
    return norm
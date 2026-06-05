import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_dot_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_modified = y + alpha * x
    tl.store(out_ptr + offsets, y_modified, mask=mask)

@triton.jit
def _dot_product_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Compute dot product
    dot = tl.sum(x * y, axis=0)
    tl.store(out_ptr, dot, mask=mask)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure inputs are on the same device and have the same dtype
    if y.device != x.device:
        x = x.to(y.device)
    if y.dtype != x.dtype:
        x = x.to(y.dtype)
    
    # Create output tensor for the modified y
    y_modified = torch.empty_like(y)
    
    # Compute the modified y: y += alpha * x
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _scaled_add_dot_kernel[grid](y, x, y_modified, n, alpha, BLOCK=block)
    
    # Compute dot product of modified y with itself
    out = torch.empty(1, dtype=y.dtype, device=y.device)
    _dot_product_kernel[grid](y_modified, y_modified, out, n, BLOCK=block)
    
    return out

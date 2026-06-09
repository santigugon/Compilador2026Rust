import torch
import triton
import triton.language as tl

@triton.jit
def _asin_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Clamp input to [-1, 1] to avoid domain errors
    x = tl.clamp(x, -1.0, 1.0)
    # Compute arcsine using the standard formula
    y = tl.atan(x / tl.sqrt(1.0 - x * x))
    tl.store(out_ptr + offsets, y, mask=mask)

def asin(input, *, out=None):
    # Handle scalar input
    if not torch.is_tensor(input):
        return torch.asin(input)
    
    # Determine output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
        if out.device != input.device:
            raise ValueError("Output tensor must be on the same device as input tensor")
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _asin_kernel[grid](input, out, n, BLOCK=block)
    return out
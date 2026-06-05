import torch
import triton
import triton.language as tl

@triton.jit
def _asin_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute arcsine using the identity: arcsin(x) = atan(x / sqrt(1 - x^2))
    # This is numerically more stable than direct computation
    y = tl.atan(x / tl.sqrt(1.0 - x * x))
    tl.store(out_ptr + offsets, y, mask=mask)

def asin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar input case
    if n == 1 and input.dim() == 0:
        input = input.unsqueeze(0)
        out = out.unsqueeze(0)
        n = 1
    
    _asin_kernel[grid](input, out, n, BLOCK=block)
    return out

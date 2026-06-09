import torch
import triton
import triton.language as tl

@triton.jit
def _asin_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute arcsine using the identity: asin(x) = atan2(x, sqrt(1 - x^2))
    one = tl.full([], 1.0, dtype=tl.float32)
    x_squared = x * x
    sqrt_one_minus_x_squared = tl.sqrt(one - x_squared)
    y = tl.atan2(x, sqrt_one_minus_x_squared)
    tl.store(out_ptr + offsets, y, mask=mask)

def asin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _asin_kernel[grid](input, out, n, BLOCK=block)
    return out

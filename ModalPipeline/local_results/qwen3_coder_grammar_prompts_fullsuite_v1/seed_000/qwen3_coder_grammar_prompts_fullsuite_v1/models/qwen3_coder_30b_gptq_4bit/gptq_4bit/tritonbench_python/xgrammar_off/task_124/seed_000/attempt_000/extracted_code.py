import torch
import triton
import triton.language as tl

@triton.jit
def _erf_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute erf using approximation: erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/π + ax^2)/(1 + ax^2)))
    # Using the approximation from Abramowitz and Stegun
    a = 0.147
    sign_x = tl.where(x >= 0, 1.0, -1.0)
    x_squared = x * x
    numerator = -x_squared * (4.0 / 3.141592653589793 + a * x_squared)
    denominator = 1.0 + a * x_squared
    exp_val = tl.exp(numerator / denominator)
    erf_val = sign_x * tl.sqrt(1.0 - exp_val)
    
    tl.store(out_ptr + offsets, erf_val, mask=mask)

def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _erf_kernel[grid](input, out, n, BLOCK=block)
    return out

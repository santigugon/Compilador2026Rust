import torch
import triton
import triton.language as tl

@triton.jit
def _erf_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Coefficients for the approximation of erf
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    # Save the sign of x
    sign = tl.where(x >= 0, 1.0, -1.0)
    x = tl.abs(x)
    
    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tl.exp(-x * x)
    
    # Apply sign
    erf_x = sign * y
    
    tl.store(out_ptr + offsets, erf_x, mask=mask)

def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _erf_kernel[grid](input, out, n, BLOCK=block)
    return out

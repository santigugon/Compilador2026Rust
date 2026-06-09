import torch
import triton
import triton.language as tl

# Constants for the rational approximation of erf
ERF_A1 = 0.254829592
ERF_A2 = -0.284496736
ERF_A3 = 1.421413741
ERF_A4 = -1.453152027
ERF_A5 = 1.061405429
ERF_P = 0.3275911

@triton.jit
def _erf_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Rational approximation for erf
    # Based on the approximation from Abramowitz and Stegun
    # erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    # where t = 1 / (1 + p*x) and p = 0.3275911
    
    # Compute t
    t = 1.0 / (1.0 + ERF_P * tl.abs(x))
    
    # Compute the polynomial
    y = ERF_A1 * t
    y = y + ERF_A2 * t * t
    y = y + ERF_A3 * t * t * t
    y = y + ERF_A4 * t * t * t * t
    y = y + ERF_A5 * t * t * t * t * t
    
    # Compute exp(-x^2)
    exp_neg_x2 = tl.exp(-x * x)
    
    # Final result
    erf_val = 1.0 - y * exp_neg_x2
    
    # Handle sign for negative x
    erf_val = tl.where(x >= 0, erf_val, -erf_val)
    
    tl.store(out_ptr + offsets, erf_val, mask=mask)


def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _erf_kernel[grid](input, out, n, BLOCK=block)
    return out
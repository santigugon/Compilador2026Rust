import torch
import triton
import triton.language as tl

@triton.jit
def _digamma_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For x <= 0, return -inf
    # For x > 0, compute digamma using asymptotic expansion
    # digamma(x) = log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    
    # Handle special case where x <= 0
    result = tl.where(x <= 0, -tl.inf, 0.0)
    
    # For positive x, compute the series expansion
    # We'll use a simplified version for better numerical stability
    # digamma(x) ≈ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    x_positive = tl.where(x > 0, x, 1.0)  # Avoid division by zero
    
    # Compute the series expansion
    log_x = tl.log(x_positive)
    term1 = 1.0 / (2.0 * x_positive)
    term2 = 1.0 / (12.0 * x_positive * x_positive)
    term3 = 1.0 / (120.0 * x_positive * x_positive * x_positive * x_positive)
    term4 = 1.0 / (252.0 * x_positive * x_positive * x_positive * x_positive * x_positive * x_positive)
    
    series = log_x - term1 - term2 + term3 - term4
    
    # Combine results
    result = tl.where(x > 0, series, -tl.inf)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def digamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _digamma_kernel[grid](input, out, n, BLOCK=block)
    return out

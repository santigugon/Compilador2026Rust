import torch
import triton
import triton.language as tl
import math

@triton.jit
def _bessel_j1_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Bessel function of the first kind of order 1
    # Using the series expansion for small x
    # For large x, we use asymptotic expansion
    # This is a simplified implementation for demonstration
    
    # Constants for the series expansion
    # J1(x) = (x/2) * sum_{n=0}^{\infty} (-1)^n * (x^2/4)^n / (n! * (n+1)!)
    
    # For small x, use series expansion
    # For large x, use asymptotic expansion
    # Here we use a simple approach that works reasonably well
    
    # Handle special cases
    x_abs = tl.abs(x)
    
    # For x = 0, J1(0) = 0
    condition = x_abs < 1e-10
    result = tl.where(condition, 0.0, 0.0)
    
    # For small x, use series expansion
    # J1(x) = x/2 * (1 - x^2/8 + x^4/192 - x^6/9216 + ...)
    # We'll use a few terms for reasonable accuracy
    x_squared = x * x
    term = x * 0.5
    sum_val = term
    term = term * (-x_squared / 8.0)
    sum_val = sum_val + term
    term = term * (-x_squared / 24.0)
    sum_val = sum_val + term
    term = term * (-x_squared / 48.0)
    sum_val = sum_val + term
    term = term * (-x_squared / 80.0)
    sum_val = sum_val + term
    
    # For larger x, use asymptotic expansion
    # This is a simplified version
    large_x = x_abs > 10.0
    # Asymptotic form: J1(x) ≈ sqrt(2/(πx)) * (cos(x - π/4) - sin(x - π/4)/x)
    # We'll use a simpler approximation for now
    result = tl.where(large_x, 
                      tl.sqrt(2.0 / (tl.pi * x_abs)) * tl.cos(x - tl.pi / 4.0),
                      sum_val)
    
    # Handle sign for negative x
    # J1(-x) = -J1(x)
    result = tl.where(x < 0, -result, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def bessel_j1(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        out = out.unsqueeze(0)
        n = 1
        grid = (1, 1)
    
    _bessel_j1_kernel[grid](input, out, n, BLOCK=block)
    
    # If input was scalar, squeeze the output
    if input.dim() == 0:
        out = out.squeeze(0)
    
    return out

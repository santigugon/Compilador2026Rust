import torch
import triton
import triton.language as tl
import math

@triton.jit
def i0_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute I0(x) using series expansion
    # I0(x) = sum_{k=0}^{\infty} (x^2/4)^k / (k!)^2
    # We'll use a truncated series for practical computation
    
    # For small x, use direct computation
    # For large x, use asymptotic approximation
    
    x = input
    x2 = x * x
    
    # Handle special case of x = 0
    is_zero = (x == 0.0)
    
    # Series expansion for small x
    result = tl.where(is_zero, 1.0, 0.0)
    
    # For non-zero x, compute series
    # We'll use a simple iterative approach with reasonable convergence
    term = tl.where(is_zero, 1.0, 1.0)
    sum_val = tl.where(is_zero, 1.0, 0.0)
    
    # Compute first few terms of the series
    for k in range(1, 20):
        term = term * x2 / (4.0 * k * k)
        sum_val = sum_val + term
        # Early termination if term becomes negligible
        if tl.abs(term) < 1e-15:
            break
    
    # For large x, use asymptotic approximation
    # I0(x) ~ e^x / sqrt(2*pi*x) for large x
    large_x = (x > 10.0)
    asymptotic = tl.exp(x) / tl.sqrt(2.0 * 3.141592653589793 * x)
    result = tl.where(large_x, asymptotic, sum_val)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    # Ensure input is float32
    if input.dtype != torch.float32:
        input = input.float()
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    i0_kernel[grid](input, out, n_elements, BLOCK_SIZE)
    
    return out

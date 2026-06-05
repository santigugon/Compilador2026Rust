import torch
import triton
import triton.language as tl
import math

@triton.jit
def _i0_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute I0(x) using the series expansion
    # I0(x) = 1 + sum_{n=1}^∞ (x^2/4)^n / (n!)^2
    
    # For numerical stability, we'll use a more robust approach
    # For small x, use direct series
    # For large x, use asymptotic expansion
    
    # Initialize result
    result = tl.ones((BLOCK,), dtype=tl.float32)
    
    # Series expansion: I0(x) = 1 + sum_{n=1}^∞ (x^2/4)^n / (n!)^2
    # We'll compute a few terms of the series
    x2 = x * x
    x2_over_4 = x2 / 4.0
    
    # Compute terms of the series
    term = x2_over_4
    n_factorial_sq = 1.0  # 0!^2 = 1
    for i in range(1, 20):  # 20 terms should be enough for convergence
        n_factorial_sq *= i * i  # n!^2
        term = term * x2_over_4 / n_factorial_sq
        result = result + term
    
    # Handle special cases
    # For x = 0, I0(0) = 1
    # For very large x, we can use asymptotic expansion
    # But for simplicity, we'll use the series for all cases
    
    tl.store(out_ptr + offsets, result, mask=mask)

def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        out = out.unsqueeze(0)
        n = 1
        grid = (1,)

    _i0_kernel[grid](input, out, n, BLOCK=block)
    return out

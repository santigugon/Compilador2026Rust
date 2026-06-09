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
    # I0(x) = 1 + sum_{n=1}^inf (x^2/4)^n / (n!)^2
    # We'll use a more numerically stable approach with iterative computation
    
    # For small x, use series expansion
    # For large x, use asymptotic expansion
    
    # Initialize result
    result = tl.ones((BLOCK,), dtype=tl.float32)
    
    # Series expansion: I0(x) = 1 + (x^2/4) + (x^2/4)^2/4 + (x^2/4)^3/36 + ...
    x2 = x * x
    term = tl.ones((BLOCK,), dtype=tl.float32)
    for i in range(1, 20):  # 20 iterations should be sufficient
        term = term * x2 / (4 * i * i)
        result = result + term
        # Early termination if term becomes negligible
        if tl.all(term < 1e-10):
            break
    
    tl.store(out_ptr + offsets, result, mask=mask)

def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _i0_kernel[grid](input, out, n, BLOCK=block)
    return out

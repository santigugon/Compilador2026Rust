import torch
import triton
import triton.language as tl

def polygamma(n, input, *, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Handle the case where n is 0 (digamma function)
    if n == 0:
        _digamma_kernel[triton.cdiv(input.numel(), 256)](input, out, input.numel(), BLOCK=256)
    else:
        # For higher derivatives, we compute the nth derivative
        _polygamma_kernel[triton.cdiv(input.numel(), 256)](input, out, n, input.numel(), BLOCK=256)
    
    return out

@triton.jit
def _digamma_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Approximation of digamma function
    # Using the asymptotic expansion for large x
    # For small x, use the recurrence relation
    y = tl.where(x > 10, tl.log(x) - 0.5 / x - 1.0 / (12 * x * x),
                 tl.log(x) - 1.0 / (2 * x) - 1.0 / (12 * x * x) + 1.0 / (360 * x * x * x * x))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _polygamma_kernel(x_ptr, out_ptr, n: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute the nth derivative of the digamma function
    # Using the formula: psi^(n)(x) = (-1)^(n+1) * n! / x^(n+1)
    # For n=1, this is the trigamma function
    # For n=2, this is the tetragamma function
    # etc.
    # We compute it directly using the formula
    result = tl.where(x > 10, 
                      (-1.0)**(n+1) * tl.math.factorial(n) / (x**(n+1)),
                      (-1.0)**(n+1) * tl.math.factorial(n) / (x**(n+1)) + 
                      (-1.0)**(n+1) * tl.math.factorial(n) / (x**(n+1)) * (1.0 / 12.0))
    tl.store(out_ptr + offsets, result, mask=mask)
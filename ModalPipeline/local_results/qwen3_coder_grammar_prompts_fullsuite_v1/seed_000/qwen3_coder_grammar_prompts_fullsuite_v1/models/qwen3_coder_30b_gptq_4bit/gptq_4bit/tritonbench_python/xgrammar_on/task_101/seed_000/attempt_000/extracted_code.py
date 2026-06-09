import torch
import triton
import triton.language as tl

def _digamma_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For numerical stability, we use the asymptotic expansion
    # digamma(x) = ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(336x^6) + ...
    # For large x, we can use the simpler approximation
    # For small x, we use the recurrence relation
    
    # Simple approximation for large x
    large_x = x > 100.0
    small_x = x <= 100.0
    
    # For large x, use asymptotic expansion
    x_large = tl.where(large_x, x, 1.0)
    log_x = tl.log(x_large)
    inv_x = 1.0 / x_large
    inv_x2 = inv_x * inv_x
    inv_x4 = inv_x2 * inv_x2
    inv_x6 = inv_x4 * inv_x2
    
    # Asymptotic expansion
    digamma_large = log_x - 0.5 * inv_x - 1.0/12.0 * inv_x2 + 1.0/120.0 * inv_x4 - 1.0/336.0 * inv_x6
    
    # For small x, use recurrence relation
    # digamma(x+1) = digamma(x) + 1/x
    # So digamma(x) = digamma(x+1) - 1/x
    # We'll compute for x in [0,1] using a lookup or approximation
    
    # Simple approximation for small x
    # Using: digamma(x) = -1/x - gamma + O(x)
    # Where gamma is Euler's constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    
    # For x = 0, return -inf
    result = tl.where(x == 0.0, -float('inf'), 
                     tl.where(small_x, 
                            -1.0/x + gamma, 
                            digamma_large))
    
    tl.store(out_ptr + offsets, result, mask=mask)


def digamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _digamma_kernel[grid](input, out, n, BLOCK=block)
    return out
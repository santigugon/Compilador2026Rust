import torch
import triton
import triton.language as tl

@triton.jit
def _i0_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute I0(x) using the series expansion
    # I0(x) = sum_{m=0}^\infty \frac{(x^2/4)^m}{(m!)^2}
    # We use a numerically stable approach with iterative computation
    
    # Initialize
    x2 = x * x
    term = 1.0
    result = 1.0
    m = 1
    
    # Iterate until convergence or max iterations
    for _ in range(100):
        term = term * x2 / (4.0 * m * m)
        result += term
        if tl.abs(term) < 1e-12:
            break
        m += 1
    
    tl.store(out_ptr + offsets, result, mask=mask)


def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
        
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _i0_kernel[grid](input, out, n, BLOCK=block)
    return out
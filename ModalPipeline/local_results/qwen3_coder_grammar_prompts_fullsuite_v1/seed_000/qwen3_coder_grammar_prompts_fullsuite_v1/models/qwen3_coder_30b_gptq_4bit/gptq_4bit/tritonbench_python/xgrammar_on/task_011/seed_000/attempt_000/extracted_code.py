import torch
import triton
import triton.language as tl

def _i0_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute I0(x) using the series expansion
    # I0(x) = sum_{k=0}^{inf} (x^2/4)^k / (k!)^2
    # We'll use a few terms for good accuracy
    
    # Initialize result
    result = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    
    # Compute terms of the series
    x2 = x * x / 4.0
    term = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    
    # Use 10 terms for good accuracy
    for k in range(1, 11):
        term = term * x2 / (k * k)
        result = result + term
    
    tl.store(out_ptr + offsets, result, mask=mask)


def i0(input, *, out=None):
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _i0_kernel[grid](input, out, n, BLOCK=block)
    return out
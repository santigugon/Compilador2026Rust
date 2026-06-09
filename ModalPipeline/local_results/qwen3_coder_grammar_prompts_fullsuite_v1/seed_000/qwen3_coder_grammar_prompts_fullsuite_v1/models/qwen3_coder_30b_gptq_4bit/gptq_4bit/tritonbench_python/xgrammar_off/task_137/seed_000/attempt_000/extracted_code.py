import torch
import triton
import triton.language as tl

@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, n: tl.constexpr, negative_slope: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(0, x) + negative_slope * tl.minimum(0, x)
    tl.store(out_ptr + offsets, y, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _leaky_relu_kernel[grid](input, out, n, negative_slope, BLOCK=block)
    return out

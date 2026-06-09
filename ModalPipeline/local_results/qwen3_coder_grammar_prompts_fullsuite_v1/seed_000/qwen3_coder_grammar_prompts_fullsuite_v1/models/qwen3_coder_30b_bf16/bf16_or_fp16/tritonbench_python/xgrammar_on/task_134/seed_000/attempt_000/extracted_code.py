import torch
import triton
import triton.language as tl

@triton.jit
def _abs_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _abs_kernel[grid](input, out, n, BLOCK=block)
    return out
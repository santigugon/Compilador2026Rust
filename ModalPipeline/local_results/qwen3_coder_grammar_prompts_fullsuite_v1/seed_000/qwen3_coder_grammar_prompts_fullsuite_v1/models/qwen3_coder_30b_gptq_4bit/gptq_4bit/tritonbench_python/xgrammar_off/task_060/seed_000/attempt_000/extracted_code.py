import torch
import triton
import triton.language as tl

@triton.jit
def _exp_sqrt_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute exp(x) followed by sqrt(exp(x)) = sqrt(exp(x)) = exp(x/2)
    y = tl.exp(x / 2.0)
    tl.store(out_ptr + offsets, y, mask=mask)

def exp_sqrt(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _exp_sqrt_kernel[grid](input, out, n, BLOCK=block)
    return out

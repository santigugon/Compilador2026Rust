import torch
import triton
import triton.language as tl

@triton.jit
def _log1p_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute log(1 + x) using the numerically stable formula
    y = tl.log1p(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _log1p_kernel[grid](input, out, n, BLOCK=block)
    return out
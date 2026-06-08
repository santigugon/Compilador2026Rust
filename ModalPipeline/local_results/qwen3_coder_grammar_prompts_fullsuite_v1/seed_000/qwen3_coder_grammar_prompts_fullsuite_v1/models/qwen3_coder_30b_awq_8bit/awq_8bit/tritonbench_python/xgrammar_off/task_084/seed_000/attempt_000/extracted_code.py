import torch
import triton
import triton.language as tl

@triton.jit
def _trunc_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Truncation for floating point: truncate towards zero
    y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    tl.store(out_ptr + offsets, y, mask=mask)

def trunc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For integer inputs, return a copy as per array-api convention
    if input.dtype.is_integral:
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out
    
    _trunc_kernel[grid](input, out, n, BLOCK=block)
    return out

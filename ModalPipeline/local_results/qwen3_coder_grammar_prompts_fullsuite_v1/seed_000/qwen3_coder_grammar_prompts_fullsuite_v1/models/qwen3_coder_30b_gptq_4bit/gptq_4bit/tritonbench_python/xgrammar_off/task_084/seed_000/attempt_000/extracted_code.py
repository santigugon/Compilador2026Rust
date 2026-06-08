import torch
import triton
import triton.language as tl

@triton.jit
def _trunc_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # For floating point, truncate towards zero
    y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    tl.store(out_ptr + offsets, y, mask=mask)

def trunc(input, *, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    # For integer inputs, return a copy
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
        return input.clone()
    
    # For floating point inputs, apply truncation
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _trunc_kernel[grid](input, out, n, BLOCK=block)
    return out

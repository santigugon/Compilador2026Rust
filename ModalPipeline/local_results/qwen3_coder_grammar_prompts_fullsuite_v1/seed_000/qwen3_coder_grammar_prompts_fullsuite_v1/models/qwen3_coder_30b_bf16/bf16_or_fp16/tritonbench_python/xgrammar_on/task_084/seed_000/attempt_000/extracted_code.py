import torch
import triton
import triton.language as tl

def _trunc_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # For floating point, truncate towards zero
    # For integer types, return the input as-is
    # We'll use a simple approach: cast to float, truncate, cast back
    # But since we're dealing with potentially integer inputs, we need to be careful
    # For simplicity, we'll assume float input and use the standard truncation
    # This is a basic implementation that works for floating point
    truncated = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    tl.store(out_ptr + offsets, truncated, mask=mask)

def trunc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
    
    # If input is already integer type, return a copy
    if input.dtype.is_integral:
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out
    
    # For floating point types, apply truncation
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _trunc_kernel[grid](input, out, n, BLOCK=block)
    return out
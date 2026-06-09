import torch
import triton
import triton.language as tl

def _trunc_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Truncation for floating point numbers
    # For integers, we just copy the value
    # We use a simple approach: cast to int and back to float
    # This works for both positive and negative numbers
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
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For integer types, we can just copy the tensor
    if input.dtype.is_integral:
        out.copy_(input)
    else:
        _trunc_kernel[grid](input, out, n, BLOCK=block)
    
    return out
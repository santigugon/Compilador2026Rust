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
    # For negative numbers, we need to be careful about the floor vs trunc behavior
    # We can use the sign and abs to handle this properly
    sign = tl.where(x >= 0, 1.0, -1.0)
    abs_x = tl.abs(x)
    truncated_abs = tl.floor(abs_x)
    truncated = sign * truncated_abs
    tl.store(out_ptr + offsets, truncated, mask=mask)

def trunc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle integer inputs by copying directly
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out
    
    _trunc_kernel[grid](input, out, n, BLOCK=block)
    return out

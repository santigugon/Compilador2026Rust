import torch
import triton
import triton.language as tl

def _get_signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # For float32, the sign bit is the most significant bit (bit 31)
    # Convert to int32 to access the sign bit
    x_bits = tl.bitcast(x, tl.int32)
    # Extract sign bit (bit 31) and convert to bool
    sign_bit = (x_bits >> 31) & 1
    # Convert to boolean (True if sign bit is set)
    result = sign_bit.to(tl.uint8) != 0
    tl.store(out_ptr + offsets, result, mask=mask)


def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != torch.bool:
            raise ValueError("Output tensor must have dtype bool")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _get_signbit_kernel[grid](input, out, n, BLOCK=block)
    return out
import torch
import triton
import triton.language as tl

@triton.jit
def _signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to integer representation to check sign bit
    x_int = tl.bitcast(x, tl.int32)
    # Extract sign bit (MSB of 32-bit float)
    sign_bit = (x_int >> 31) & 1
    # Convert back to boolean
    result = sign_bit.to(tl.uint8) != 0
    tl.store(out_ptr + offsets, result, mask=mask)

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == torch.bool, "Output tensor must have bool dtype"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _signbit_kernel[grid](input, out, n, BLOCK=block)
    return out

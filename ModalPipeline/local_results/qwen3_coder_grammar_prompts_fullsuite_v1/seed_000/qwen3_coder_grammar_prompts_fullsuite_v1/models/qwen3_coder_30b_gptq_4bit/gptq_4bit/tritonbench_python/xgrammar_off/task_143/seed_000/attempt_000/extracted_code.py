import torch
import triton
import triton.language as tl

@triton.jit
def _signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Check if the sign bit is set (negative numbers or negative zero)
    # For IEEE 754 float32/64, the sign bit is the most significant bit
    # We can use bit manipulation to check this
    x_bits = tl.cast_to_shared(x, tl.int32)
    sign_bit = (x_bits >> 31) & 1
    tl.store(out_ptr + offsets, sign_bit, mask=mask)

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    else:
        assert out.dtype == torch.bool, "Output tensor must have bool dtype"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle different dtypes
    if input.dtype in [torch.float32, torch.float64]:
        _signbit_kernel[grid](input, out, n, BLOCK=block)
    else:
        # For non-floating point types, use PyTorch's implementation
        # This is a fallback for integer types or other dtypes
        out = torch.signbit(input)
    
    return out

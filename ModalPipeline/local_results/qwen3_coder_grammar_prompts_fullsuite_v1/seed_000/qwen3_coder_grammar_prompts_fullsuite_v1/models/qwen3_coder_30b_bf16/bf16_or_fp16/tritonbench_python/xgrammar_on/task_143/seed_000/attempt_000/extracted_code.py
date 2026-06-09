import torch
import triton
import triton.language as tl

def _get_signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Convert to unsigned integer representation to check sign bit
    x_bits = tl.bitcast(x, tl.uint32)
    # Extract sign bit (MSB of 32-bit float)
    sign_bit = (x_bits >> 31) & 1
    # Convert back to boolean
    result = sign_bit.to(tl.uint8) != 0
    tl.store(out_ptr + offsets, result, mask=mask)


def _get_signbit_kernel_64(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Convert to unsigned integer representation to check sign bit
    x_bits = tl.bitcast(x, tl.uint64)
    # Extract sign bit (MSB of 64-bit float)
    sign_bit = (x_bits >> 63) & 1
    # Convert back to boolean
    result = sign_bit.to(tl.uint8) != 0
    tl.store(out_ptr + offsets, result, mask=mask)


def signbit(input, *, out=None):
    if out is None:
        out = torch.empty(input.shape, dtype=torch.bool, device=input.device)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if input.dtype == torch.float32:
        _get_signbit_kernel[grid](input, out, n, BLOCK=block)
    elif input.dtype == torch.float64:
        _get_signbit_kernel_64[grid](input, out, n, BLOCK=block)
    else:
        # For non-floating point types, use PyTorch implementation
        return torch.signbit(input)
    
    return out
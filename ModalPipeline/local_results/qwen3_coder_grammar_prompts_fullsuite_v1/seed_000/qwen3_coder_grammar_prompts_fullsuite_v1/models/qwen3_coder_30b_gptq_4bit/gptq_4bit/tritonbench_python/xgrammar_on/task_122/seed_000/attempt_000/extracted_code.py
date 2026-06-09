import torch
import triton
import triton.language as tl

def _broadcast_shape(input_shape, exponent_shape):
    # Simple broadcast shape check
    if len(input_shape) == len(exponent_shape):
        return input_shape
    elif len(input_shape) > len(exponent_shape):
        return input_shape
    else:
        return exponent_shape

@triton.jit
def _pow_kernel(input_ptr, exponent_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    exponent = tl.load(exponent_ptr + offsets, mask=mask, other=0.0)
    result = tl.power(input, exponent)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _pow_scalar_kernel(input_ptr, out_ptr, exponent: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.power(input, exponent)
    tl.store(out_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if not torch.is_tensor(exponent):
        # Scalar exponent case
        _pow_scalar_kernel[grid](input, out, exponent, n, BLOCK=block)
    else:
        # Tensor exponent case
        if exponent.numel() == 1:
            # Single element tensor
            exponent_scalar = exponent.item()
            _pow_scalar_kernel[grid](input, out, exponent_scalar, n, BLOCK=block)
        else:
            # Full tensor exponent
            _pow_kernel[grid](input, exponent, out, n, BLOCK=block)
    
    return out
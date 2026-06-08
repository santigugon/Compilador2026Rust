import torch
import triton
import triton.language as tl

@triton.jit
def _pow_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = tl.power(x, y)
    tl.store(out_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    # Handle scalar exponent case
    if not torch.is_tensor(exponent):
        # Convert scalar to tensor with same shape as input
        exponent = torch.tensor(exponent, dtype=input.dtype, device=input.device)
    
    # Ensure exponent has the same number of elements as input
    if exponent.numel() != input.numel():
        # If exponent is a scalar tensor, broadcast it
        if exponent.numel() == 1:
            exponent = exponent.expand_as(input)
        else:
            raise ValueError("exponent tensor must have the same number of elements as input tensor or be a scalar")
    
    # Handle broadcasting
    out_shape = torch.broadcast_tensors(input, exponent)[0].shape
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _pow_kernel[grid](input, exponent, out, n, BLOCK=block)
    return out

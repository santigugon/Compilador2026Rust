import torch
import triton
import triton.language as tl

@triton.jit
def _tanh_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh using the numerically stable formula
    y = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0
    tl.store(out_ptr + offsets, y, mask=mask)


def tanh(input, *, out=None):
    if not torch.is_tensor(input):
        # Fall back to PyTorch for scalar inputs
        return torch.tanh(input)
    
    # Handle the case where out is provided
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
    _tanh_kernel[grid](input, out, n, BLOCK=block)
    return out
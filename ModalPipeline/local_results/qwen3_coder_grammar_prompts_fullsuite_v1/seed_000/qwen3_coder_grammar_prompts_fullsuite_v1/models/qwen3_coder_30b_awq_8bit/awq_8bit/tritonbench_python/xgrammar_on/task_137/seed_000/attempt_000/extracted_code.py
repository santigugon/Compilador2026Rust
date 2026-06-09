import torch
import triton
import triton.language as tl

@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, n: tl.constexpr, negative_slope: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute max(0, x) + negative_slope * min(0, x)
    positive_part = tl.where(x > 0, x, 0.0)
    negative_part = tl.where(x < 0, x * negative_slope, 0.0)
    result = positive_part + negative_part
    tl.store(out_ptr + offsets, result, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        # For inplace operation, we modify the input tensor directly
        out = input
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _leaky_relu_kernel[grid](input, input, n, negative_slope, BLOCK=block)
    else:
        # For non-inplace operation, we create a new output tensor
        out = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _leaky_relu_kernel[grid](input, out, n, negative_slope, BLOCK=block)
    return out
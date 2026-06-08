import torch
import triton
import triton.language as tl

@triton.jit
def _relu_sqrt_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU: set negative values to zero
    x = tl.maximum(x, 0.0)
    # Apply square root
    y = tl.sqrt(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    if inplace:
        # For inplace operation, we need to ensure the input tensor can be modified
        # If out is provided, it should be the same as input for inplace operation
        if out is not None and out is not input:
            raise ValueError("For inplace operation, out must be the same tensor as input")
        out = input
    else:
        # If out is provided, use it; otherwise, create a new tensor
        if out is not None:
            out = out
        else:
            out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _relu_sqrt_kernel[grid](input, out, n, BLOCK=block)
    return out

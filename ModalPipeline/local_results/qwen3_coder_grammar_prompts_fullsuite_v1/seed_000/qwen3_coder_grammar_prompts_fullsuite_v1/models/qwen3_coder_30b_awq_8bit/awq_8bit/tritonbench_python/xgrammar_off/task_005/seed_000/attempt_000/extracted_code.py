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
        # For inplace operation, we need to ensure the input is contiguous
        # and modify it in place
        if not input.is_contiguous():
            input = input.contiguous()
        # Apply ReLU and sqrt in place
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _relu_sqrt_kernel[grid](input, input, n, BLOCK=block)
        return input
    else:
        # For non-inplace operation, create output tensor
        if out is not None:
            # Use provided output tensor
            out = torch.empty_like(out)
        else:
            # Create new output tensor
            out = torch.empty_like(input)
        
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _relu_sqrt_kernel[grid](input, out, n, BLOCK=block)
        return out

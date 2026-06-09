import torch
import triton
import triton.language as tl

def relu_sqrt(input, inplace=False, out=None):
    if out is not None:
        output = out
    elif inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    if inplace:
        # For in-place operation, we need to ensure the tensor is contiguous
        if not input.is_contiguous():
            input = input.contiguous()
        output = input
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Use a kernel that applies ReLU and then square root
    _relu_sqrt_kernel[grid](input, output, n, BLOCK=block)
    
    return output

@triton.jit
def _relu_sqrt_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU: max(0, x)
    x_relu = tl.maximum(0.0, x)
    # Apply square root
    y = tl.sqrt(x_relu)
    tl.store(out_ptr + offsets, y, mask=mask)
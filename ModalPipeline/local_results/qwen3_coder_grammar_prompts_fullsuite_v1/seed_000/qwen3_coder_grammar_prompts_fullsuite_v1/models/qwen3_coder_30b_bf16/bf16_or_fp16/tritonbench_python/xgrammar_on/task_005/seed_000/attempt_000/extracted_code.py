import torch
import triton
import triton.language as tl

@triton.jit
def _relu_sqrt_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU: max(0, x)
    x_relu = tl.maximum(x, 0.0)
    # Apply square root
    y = tl.sqrt(x_relu)
    tl.store(y_ptr + offsets, y, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    # Handle scalar input
    if input.dim() == 0:
        if inplace:
            input = input.clone()  # Can't modify scalar in-place
        return torch.sqrt(torch.relu(input))
    
    # Determine output tensor
    if out is not None:
        if inplace:
            raise ValueError("Cannot specify both 'out' and 'inplace=True'")
        output = out
    elif inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Get number of elements
    n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    _relu_sqrt_kernel[grid](input, output, n, BLOCK=block)
    
    return output
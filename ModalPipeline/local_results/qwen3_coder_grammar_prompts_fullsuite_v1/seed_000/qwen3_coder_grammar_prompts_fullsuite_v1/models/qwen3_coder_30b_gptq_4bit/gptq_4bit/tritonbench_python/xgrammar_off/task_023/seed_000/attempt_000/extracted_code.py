import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, Y, n_elements, BLOCK_SIZE: int = 1024):
    """Element-wise ReLU operation."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(Y + offsets, y, mask=mask)

def relu(input, inplace=False):
    """Apply ReLU activation function element-wise."""
    if not inplace:
        output = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        relu_kernel[grid](input, output, n_elements)
        return output
    else:
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        relu_kernel[grid](input, input, n_elements)
        return input

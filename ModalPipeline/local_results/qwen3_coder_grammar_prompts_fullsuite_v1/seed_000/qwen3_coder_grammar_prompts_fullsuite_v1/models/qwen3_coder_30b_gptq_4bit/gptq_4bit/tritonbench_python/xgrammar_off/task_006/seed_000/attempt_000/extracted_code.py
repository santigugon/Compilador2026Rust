import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_kernel(X, Y, n_elements, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.sqrt(x)
    tl.store(Y + offsets, y, mask=mask)

def sqrt(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    sqrt_kernel[grid](input, out, n_elements)
    
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def cos_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.cos(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def cos(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    cos_kernel[grid](input, out, n_elements)
    
    return out

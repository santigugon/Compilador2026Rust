import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.log(1.0 + x)
    tl.store(y_ptr + offsets, y, mask=mask)

def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
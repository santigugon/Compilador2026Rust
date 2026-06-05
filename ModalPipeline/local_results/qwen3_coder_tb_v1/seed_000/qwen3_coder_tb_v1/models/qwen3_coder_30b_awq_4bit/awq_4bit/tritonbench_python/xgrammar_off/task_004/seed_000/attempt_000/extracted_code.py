import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(X, Y, n_elements, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.tanh(x)
    tl.store(Y + offsets, y, mask=mask)

def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(X, Y, n_elements, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.log(x)
    tl.store(Y + offsets, y, mask=mask)

def log(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    
    if input.numel() == 0:
        return out
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    log_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out

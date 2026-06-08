import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(X, Y, N, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.rsqrt(x)
    tl.store(Y + offsets, y, mask=mask)

def rsqrt(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    
    if input.numel() == 0:
        return out
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    rsqrt_kernel[grid](input, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

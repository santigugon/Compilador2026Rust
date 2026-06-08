import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.log1p(x)
    tl.store(Y + offsets, y, mask=mask)

def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    else:
        if out.shape != input.shape or out.dtype != input.dtype or out.device != input.device:
            raise ValueError("out tensor must have the same shape, dtype, and device as input tensor")
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    log1p_kernel[grid](input, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

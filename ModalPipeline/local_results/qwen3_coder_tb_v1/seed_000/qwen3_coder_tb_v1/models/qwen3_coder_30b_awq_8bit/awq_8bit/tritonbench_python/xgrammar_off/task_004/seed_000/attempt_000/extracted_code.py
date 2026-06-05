import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.tanh(x)
    tl.store(Y + offsets, y, mask=mask)

def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    assert input.dtype == torch.float32, "Only float32 inputs are supported"
    assert out.dtype == torch.float32, "Only float32 outputs are supported"
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    tanh_kernel[grid](input, out, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

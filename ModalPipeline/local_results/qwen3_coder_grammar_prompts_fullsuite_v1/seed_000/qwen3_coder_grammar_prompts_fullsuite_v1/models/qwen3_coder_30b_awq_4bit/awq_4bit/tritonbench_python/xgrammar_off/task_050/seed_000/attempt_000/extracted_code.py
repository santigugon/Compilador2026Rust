import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_exp_kernel(X, Y, N, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.exp(tl.sqrt(x))
    tl.store(Y + offsets, y, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)
    
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    
    N = input.numel()
    grid = (triton.cdiv(N, 1024),)
    
    sqrt_exp_kernel[grid](input, out, N, BLOCK_SIZE=1024)
    
    return out

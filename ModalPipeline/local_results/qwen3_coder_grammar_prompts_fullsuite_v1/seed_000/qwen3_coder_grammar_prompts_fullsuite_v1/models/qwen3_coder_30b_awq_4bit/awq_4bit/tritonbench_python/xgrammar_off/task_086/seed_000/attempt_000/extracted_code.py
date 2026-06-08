import torch
import triton
import triton.language as tl

@triton.jit
def log_tanh_kernel(X, Y, N, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.tanh(tl.log(x))
    tl.store(Y + offsets, y, mask=mask)

def log_tanh(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    assert input.dtype == torch.float32, "Input must be float32"
    assert input.device == out.device, "Input and output must be on the same device"
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    log_tanh_kernel[grid](input, out, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

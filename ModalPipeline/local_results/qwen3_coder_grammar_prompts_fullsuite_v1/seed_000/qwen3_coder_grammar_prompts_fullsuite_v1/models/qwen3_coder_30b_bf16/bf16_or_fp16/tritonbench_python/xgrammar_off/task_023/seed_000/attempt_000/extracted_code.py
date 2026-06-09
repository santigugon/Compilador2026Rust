import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(Y + offsets, y, mask=mask)

def relu(input, inplace=False):
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input
    
    if input.numel() == 0:
        return output
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    relu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output

import torch
import triton
import triton.language as tl

@triton.jit
def exp_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.sqrt(tl.exp(input))
    tl.store(output_ptr + offsets, output, mask=mask)

def exp_sqrt(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    exp_sqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE)
    
    return out

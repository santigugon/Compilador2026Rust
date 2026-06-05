import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.math.sqrt(input)
    output = tl.math.tanh(output)
    tl.store(output_ptr + offsets, output, mask=mask)

def sqrt_tanh(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    sqrt_tanh_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

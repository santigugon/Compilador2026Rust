import torch
import triton
import triton.language as tl

@triton.jit
def asin_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.asin(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def asin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)
    
    if input.is_cuda:
        n_elements = input.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        asin_kernel[grid](
            input_ptr=input,
            output_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        out = torch.asin(input)
    
    return out

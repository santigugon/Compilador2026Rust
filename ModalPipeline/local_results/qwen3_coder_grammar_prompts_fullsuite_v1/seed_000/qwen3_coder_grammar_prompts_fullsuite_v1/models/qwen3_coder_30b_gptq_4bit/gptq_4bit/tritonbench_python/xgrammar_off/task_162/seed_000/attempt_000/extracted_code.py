import triton
import triton.language as tl
import torch

@triton.jit
def reciprocal_kernel(
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
    output = 1.0 / input
    tl.store(output_ptr + offsets, output, mask=mask)

def reciprocal(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    if input.dtype in [torch.int32, torch.int64]:
        input = input.to(torch.float32)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    reciprocal_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def _polygamma_kernel(n, input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.libdevice.tl_polygamma(n, input)
    tl.store(output_ptr + offsets, output, mask=mask)

def polygamma(n, input, *, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    if input.dtype != torch.float32:
        input = input.float()
    
    if out.dtype != torch.float32:
        out = out.float()
    
    num_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    _polygamma_kernel[grid](
        n,
        input,
        out,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

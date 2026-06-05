import torch
import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
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
    output = tl.where(input >= 0, tl.floor(input), tl.ceil(input))
    tl.store(output_ptr + offsets, output, mask=mask)

def trunc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
        if out.device != input.device:
            raise ValueError("Output tensor must be on the same device as input tensor")
    
    if input.numel() == 0:
        return out
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    trunc_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = tl.where(input_vals < 0.0, 1, 0)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool, device=input.device)
    else:
        if out.dtype != torch.bool:
            raise ValueError("Output tensor must have bool dtype")
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input")
    
    num_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    signbit_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

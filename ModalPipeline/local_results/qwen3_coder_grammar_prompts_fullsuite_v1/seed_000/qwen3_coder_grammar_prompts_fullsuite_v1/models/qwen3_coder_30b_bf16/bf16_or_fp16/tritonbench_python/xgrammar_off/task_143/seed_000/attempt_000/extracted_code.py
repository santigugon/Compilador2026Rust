import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(input < 0.0, 1, 0)
    tl.store(output_ptr + offsets, output, mask=mask)

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool, device=input.device)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != torch.bool:
            raise ValueError("Output tensor must have boolean dtype")
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    signbit_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

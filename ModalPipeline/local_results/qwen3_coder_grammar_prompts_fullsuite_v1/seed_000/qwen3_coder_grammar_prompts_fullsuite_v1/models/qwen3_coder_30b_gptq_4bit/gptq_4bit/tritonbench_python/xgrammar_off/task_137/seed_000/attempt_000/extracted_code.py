import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(input > 0, input, negative_slope * input)
    tl.store(output_ptr + offsets, output, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    leaky_relu_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        negative_slope=negative_slope,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

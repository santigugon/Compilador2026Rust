import torch
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    alpha: float,
    scale: float,
    BLOCK_SIZE: int = 1024,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    exp_input = tl.exp(input)
    selu_val = scale * tl.where(input >= 0, input, alpha * (exp_input - 1))
    tl.store(output_ptr + offsets, selu_val, mask=mask)

def selu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    alpha = 1.6732632423543772848117116294564
    scale = 1.0507009873554804934193349852946
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    selu_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        alpha=alpha,
        scale=scale,
        BLOCK_SIZE=1024
    )
    
    return output

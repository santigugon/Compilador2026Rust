import torch
import triton
import triton.language as tl

@triton.jit
def relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    relu_input = tl.where(input > 0.0, input, 0.0)
    output = tl.sqrt(relu_input)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    if inplace and input.is_contiguous():
        if out is not None:
            raise ValueError("Cannot specify both 'inplace' and 'out'")
        output = input
    else:
        if out is not None:
            output = out
        else:
            output = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    relu_sqrt_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

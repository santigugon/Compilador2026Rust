import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    approximate,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    if approximate == 1:
        # Approximate GELU using tanh
        output = 0.5 * input * (1 + tl.tanh(0.7978845608 * (input + 0.044715 * input * input * input)))
    else:
        # Exact GELU using erf
        output = 0.5 * input * (1 + tl.erf(input / tl.sqrt(2.0)))
    
    tl.store(output_ptr + offsets, output, mask=mask)

def gelu(input, approximate='none'):
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    input = input.contiguous()
    output = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    approximate_flag = 1 if approximate == 'tanh' else 0
    
    gelu_kernel[grid](
        input,
        output,
        n_elements,
        approximate_flag,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

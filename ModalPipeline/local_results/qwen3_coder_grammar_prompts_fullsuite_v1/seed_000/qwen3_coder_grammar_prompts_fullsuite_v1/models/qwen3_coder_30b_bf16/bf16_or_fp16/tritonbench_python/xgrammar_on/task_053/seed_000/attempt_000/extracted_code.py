import torch
import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(input_ptr, other_ptr, output_ptr, n_elements, inplace, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    
    output = input * other
    output = tl.maximum(output, 0.0)
    
    if inplace:
        tl.store(input_ptr + offsets, output, mask=mask)
    else:
        tl.store(output_ptr + offsets, output, mask=mask)

def mul_relu(input, other, inplace=False, out=None) -> torch.Tensor:
    if inplace and out is not None:
        raise ValueError("Cannot specify both 'inplace=True' and 'out' parameter.")
    
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    if out is None:
        out = torch.empty_like(input)
    
    if inplace:
        out = input
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    mul_relu_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        inplace=inplace,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
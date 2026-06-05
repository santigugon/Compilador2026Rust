import torch
import triton
import triton.language as tl

@triton.jit
def _bitwise_and_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    output_vals = input_vals & other_vals
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def bitwise_and(input, other, *, out=None):
    if input.dtype not in [torch.bool, torch.int32, torch.int64, torch.int16, torch.int8]:
        raise TypeError("Input tensor must be of integral or Boolean type")
    if other.dtype not in [torch.bool, torch.int32, torch.int64, torch.int16, torch.int8]:
        raise TypeError("Other tensor must be of integral or Boolean type")
    if input.shape != other.shape:
        raise ValueError("Input tensors must have the same shape")
    
    output = torch.empty_like(input, dtype=torch.bool if input.dtype == torch.bool else input.dtype)
    if out is not None:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        output = out
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _bitwise_and_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE
    )
    
    return output

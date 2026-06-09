import torch
import triton
import triton.language as tl

@triton.jit
def _bitwise_and_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
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
    
    size = input.numel()
    output = torch.empty_like(input) if out is None else out
    
    if size == 0:
        return output
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    _bitwise_and_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        output.data_ptr(),
        size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

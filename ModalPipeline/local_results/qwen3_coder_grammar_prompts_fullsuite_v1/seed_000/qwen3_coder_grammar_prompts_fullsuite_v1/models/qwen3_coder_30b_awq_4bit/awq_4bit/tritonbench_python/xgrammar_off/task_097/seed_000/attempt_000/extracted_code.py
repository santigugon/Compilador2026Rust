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
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool if input.dtype == torch.bool else input.dtype)
    
    if input.dtype != other.dtype:
        raise ValueError("input and other must have the same dtype")
    
    if input.dtype not in [torch.bool, torch.int32, torch.int64]:
        raise ValueError("input must be of integral or Boolean types")
    
    if other.dtype not in [torch.bool, torch.int32, torch.int64]:
        raise ValueError("other must be of integral or Boolean types")
    
    size = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    _bitwise_and_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

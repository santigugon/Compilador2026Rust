import torch
import triton
import triton.language as tl

@triton.jit
def _ifftshift_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < dim_size
    input_offsets = indices * stride
    output_offsets = indices * stride
    tl.store(output_ptr + output_offsets, tl.load(input_ptr + input_offsets), mask=mask)

def ifftshift(input, dim=None):
    if dim is None:
        # Handle all dimensions
        input = input.clone()
        for i in range(len(input.shape)):
            input = _ifftshift_single_dim(input, i)
        return input
    else:
        # Handle specific dimensions
        if isinstance(dim, int):
            dim = [dim]
        input = input.clone()
        for d in dim:
            input = _ifftshift_single_dim(input, d)
        return input

def _ifftshift_single_dim(input, dim):
    shape = input.shape
    dim_size = shape[dim]
    stride = 1
    for i in range(dim):
        stride *= shape[i]
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Calculate block size and number of blocks
    BLOCK_SIZE = 1024
    num_blocks = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    _ifftshift_kernel[(num_blocks,)](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        dim_size=dim_size,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply shift logic
    if dim_size % 2 == 0:
        # Even size: shift by half
        half = dim_size // 2
        if dim == 0:
            output = torch.cat([input[..., half:], input[..., :half]], dim=dim)
        else:
            slices = [slice(None)] * len(shape)
            slices[dim] = slice(half, None)
            part1 = input[tuple(slices)]
            slices[dim] = slice(None, half)
            part2 = input[tuple(slices)]
            output = torch.cat([part1, part2], dim=dim)
    else:
        # Odd size: shift by half (integer division)
        half = dim_size // 2
        if dim == 0:
            output = torch.cat([input[..., half+1:], input[..., :half+1]], dim=dim)
        else:
            slices = [slice(None)] * len(shape)
            slices[dim] = slice(half+1, None)
            part1 = input[tuple(slices)]
            slices[dim] = slice(None, half+1)
            part2 = input[tuple(slices)]
            output = torch.cat([part1, part2], dim=dim)
    
    return output

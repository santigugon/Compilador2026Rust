import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    dim_size: tl.constexpr,
    other_dims_size: tl.constexpr,
    stride_input_dim: tl.constexpr,
    stride_output_dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate which slice we're working on
    slice_id = pid // dim_size
    dim_id = pid % dim_size
    
    # Calculate the offset for this slice
    slice_offset = slice_id * other_dims_size
    
    # Load the input data for this slice
    input_offsets = slice_offset + dim_id * stride_input_dim
    output_offsets = slice_offset + dim_id * stride_output_dim
    
    # Load the input value
    input_val = tl.load(input_ptr + input_offsets, mask=dim_id < dim_size)
    
    # Apply softmax computation
    # For numerical stability, subtract the maximum value
    max_val = tl.max(input_val, axis=0)
    exp_val = tl.exp(input_val - max_val)
    sum_val = tl.sum(exp_val, axis=0)
    softmax_val = exp_val / sum_val
    
    # Store the result
    tl.store(output_ptr + output_offsets, softmax_val, mask=dim_id < dim_size)

def softmax(input, dim, dtype=None):
    # Handle dtype casting if needed
    if dtype is not None:
        input = input.to(dtype)
    
    # Get the shape and strides
    shape = input.shape
    strides = input.stride()
    
    # Calculate the size of the specified dimension
    dim_size = shape[dim]
    
    # Calculate the size of all other dimensions
    other_dims_size = 1
    for i in range(len(shape)):
        if i != dim:
            other_dims_size *= shape[i]
    
    # Calculate strides for the specified dimension
    stride_input_dim = strides[dim]
    stride_output_dim = strides[dim]
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Determine block size and grid size
    BLOCK = 256
    grid = (other_dims_size * dim_size + BLOCK - 1) // BLOCK
    
    # Launch kernel
    _softmax_kernel[grid](
        input,
        output,
        dim_size,
        other_dims_size,
        stride_input_dim,
        stride_output_dim,
        BLOCK=BLOCK
    )
    
    return output

import torch
import triton
import triton.language as tl

def _softmax_kernel(input_ptr, output_ptr, stride_input, stride_output, dim_size, num_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Calculate the number of elements per block
    num_blocks = tl.cdiv(num_elements, BLOCK)
    
    # Each block handles one element along the specified dimension
    if pid >= num_blocks:
        return
    
    # Calculate the offset for this block
    offset = pid * BLOCK
    
    # Load data from input tensor
    input_offsets = offset + tl.arange(0, BLOCK)
    mask = input_offsets < num_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply softmax
    # For numerical stability, subtract the maximum value
    max_val = tl.max(input_vals, axis=0)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store results
    tl.store(output_ptr + input_offsets, softmax_vals, mask=mask)

@triton.jit
def _softmax_kernel_2d(input_ptr, output_ptr, stride_input_row, stride_input_col, stride_output_row, stride_output_col, 
                       rows, cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return
    
    # Load a row from input
    input_row_offsets = row * stride_input_row + tl.arange(0, BLOCK_SIZE)
    input_vals = tl.load(input_ptr + input_row_offsets, mask=tl.arange(0, BLOCK_SIZE) < cols, other=0.0)
    
    # Compute softmax for this row
    max_val = tl.max(input_vals, axis=0)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    output_row_offsets = row * stride_output_row + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + output_row_offsets, softmax_vals, mask=tl.arange(0, BLOCK_SIZE) < cols)

@triton.jit
def _softmax_kernel_1d(input_ptr, output_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Numerical stability
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_x = exp_x / sum_exp
    
    tl.store(output_ptr + offsets, softmax_x, mask=mask)

@triton.jit
def _softmax_kernel_dim(input_ptr, output_ptr, stride_input, stride_output, dim_size, num_elements, 
                        dim_stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which slice we're processing
    slice_id = pid // dim_size
    element_id = pid % dim_size
    
    if slice_id >= num_elements // dim_size:
        return
    
    # Calculate the offset for this slice
    slice_offset = slice_id * dim_stride
    
    # Load the slice
    input_offsets = slice_offset + tl.arange(0, dim_size)
    input_vals = tl.load(input_ptr + input_offsets, mask=tl.arange(0, dim_size) < dim_size, other=0.0)
    
    # Compute softmax
    max_val = tl.max(input_vals, axis=0)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    output_offsets = slice_offset + tl.arange(0, dim_size)
    tl.store(output_ptr + output_offsets, softmax_vals, mask=tl.arange(0, dim_size) < dim_size)

@triton.jit
def _softmax_kernel_general(input_ptr, output_ptr, strides_input, strides_output, dim_size, num_elements, 
                            dim_stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which slice we're processing
    slice_id = pid // dim_size
    element_id = pid % dim_size
    
    if slice_id >= num_elements // dim_size:
        return
    
    # Calculate the offset for this slice
    slice_offset = slice_id * dim_stride
    
    # Load the slice
    input_offsets = slice_offset + tl.arange(0, dim_size)
    input_vals = tl.load(input_ptr + input_offsets, mask=tl.arange(0, dim_size) < dim_size, other=0.0)
    
    # Compute softmax
    max_val = tl.max(input_vals, axis=0)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    output_offsets = slice_offset + tl.arange(0, dim_size)
    tl.store(output_ptr + output_offsets, softmax_vals, mask=tl.arange(0, dim_size) < dim_size)

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle negative dimensions
    if dim < 0:
        dim = input.dim() + dim
    
    # Special case: 1D tensor
    if input.dim() == 1:
        out = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _softmax_kernel_1d[grid](input, out, n, BLOCK=block)
        return out
    
    # For 2D tensors, use optimized kernel
    if input.dim() == 2:
        out = torch.empty_like(input)
        rows, cols = input.shape
        block_size = 256
        grid = (rows,)
        _softmax_kernel_2d[grid](input, out, input.stride(0), input.stride(1), 
                                out.stride(0), out.stride(1), 
                                rows, cols, BLOCK_SIZE=block_size)
        return out
    
    # For higher dimensions, we need to handle the specific dimension
    # Get the size of the specified dimension
    dim_size = input.shape[dim]
    
    # Calculate total number of elements
    num_elements = input.numel()
    
    # Calculate stride for the specified dimension
    dim_stride = input.stride(dim)
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Use a simple approach for now - launch one kernel per slice
    block = 256
    grid = (triton.cdiv(num_elements, block),)
    
    # For simplicity, we'll use a general kernel approach
    _softmax_kernel_general[grid](input, out, input.stride(0) if input.dim() > 0 else 0, 
                                out.stride(0) if out.dim() > 0 else 0, 
                                dim_size, num_elements, dim_stride, BLOCK=block)
    
    return out
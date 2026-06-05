import torch
import triton
import triton.language as tl

@triton.jit
def _min_kernel(x_ptr, out_ptr, indices_ptr, n_rows, n_cols, stride_x_row, stride_x_col, stride_out_row, stride_out_col, stride_indices_row, stride_indices_col, keepdim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    
    # Load the row data
    row_offsets = pid * stride_x_row
    row_ptr = x_ptr + row_offsets
    
    # Initialize min and index
    min_val = tl.load(row_ptr + tl.arange(0, BLOCK_SIZE) * stride_x_col)
    min_val = tl.min(min_val)
    
    # Find the index of the first minimum value
    indices = tl.arange(0, BLOCK_SIZE) * stride_x_col
    min_indices = tl.argmin(min_val, indices)
    
    # Store results
    out_row_offset = pid * stride_out_row
    indices_row_offset = pid * stride_indices_row
    
    tl.store(out_ptr + out_row_offset, min_val)
    tl.store(indices_ptr + indices_row_offset, min_indices)

def min(input, dim, keepdim=False, *, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    if dim < 0:
        dim = input.dim() + dim
    
    if dim < 0 or dim >= input.dim():
        raise ValueError("dim out of range")
    
    # Get dimensions
    input_shape = input.shape
    n_rows = 1
    n_cols = input_shape[dim]
    
    # Calculate strides
    input_strides = input.stride()
    stride_x_row = input_strides[dim] if dim < len(input_strides) else 1
    stride_x_col = input_strides[0] if len(input_strides) > 0 else 1
    
    # Calculate output shape
    output_shape = list(input_shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    
    # Create output tensors
    out_min = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    out_indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Calculate grid
    n_rows = input_shape[0] if dim == 0 else 1
    for i in range(dim):
        n_rows *= input_shape[i]
    
    block_size = 256
    grid = (triton.cdiv(n_rows, block_size),)
    
    # Launch kernel
    if n_rows > 0:
        _min_kernel[grid](
            input, 
            out_min, 
            out_indices, 
            n_rows, 
            n_cols, 
            stride_x_row, 
            stride_x_col, 
            out_min.stride()[0] if len(out_min.stride()) > 0 else 1, 
            out_min.stride()[1] if len(out_min.stride()) > 1 else 1, 
            out_indices.stride()[0] if len(out_indices.stride()) > 0 else 1, 
            out_indices.stride()[1] if len(out_indices.stride()) > 1 else 1, 
            keepdim, 
            block_size
        )
    
    return (out_min, out_indices)

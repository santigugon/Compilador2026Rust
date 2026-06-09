import torch
import triton
import triton.language as tl

def _min_row_kernel(input_ptr, output_ptr, indices_ptr, rows, cols, stride_input_row, stride_input_col, stride_output_row, stride_output_col, stride_indices_row, stride_indices_col, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    
    # Initialize min value and index
    min_val = tl.full([1], float('inf'), dtype=tl.float32)
    min_idx = tl.full([1], 0, dtype=tl.int64)
    
    # Load a row of input
    input_row_ptr = input_ptr + pid * stride_input_row
    
    # Process in blocks
    for i in range(0, cols, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < cols
        
        # Load input values
        input_vals = tl.load(input_row_ptr + offsets * stride_input_col, mask=mask, other=float('inf'))
        
        # Find min in this block
        block_min_val = tl.min(input_vals)
        block_min_idx = tl.argmin(input_vals)
        
        # Update global min
        if block_min_val < min_val:
            min_val = block_min_val
            min_idx = block_min_idx + i
    
    # Store results
    if keepdim:
        tl.store(output_ptr + pid * stride_output_row, min_val)
        tl.store(indices_ptr + pid * stride_indices_row, min_idx)
    else:
        tl.store(output_ptr + pid, min_val)
        tl.store(indices_ptr + pid, min_idx)


def min(input, dim, keepdim=False, *, out=None):
    if out is not None:
        output, indices = out
    else:
        # Create output tensors
        input_shape = input.shape
        if keepdim:
            output_shape = input_shape
            output_shape[dim] = 1
        else:
            output_shape = list(input_shape)
            output_shape.pop(dim)
        
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Handle negative dim
    if dim < 0:
        dim = input.dim() + dim
    
    # Get dimensions
    rows = 1
    cols = input.shape[dim]
    for i in range(dim):
        rows *= input.shape[i]
    for i in range(dim + 1, input.dim()):
        cols *= input.shape[i]
    
    # Get strides
    stride_input_row = input.stride(dim) if dim < input.dim() - 1 else 1
    stride_input_col = input.stride(0) if dim == 0 else input.stride(dim)
    
    stride_output_row = output.stride(0) if output.dim() > 0 else 1
    stride_output_col = output.stride(0) if output.dim() > 0 else 1
    
    stride_indices_row = indices.stride(0) if indices.dim() > 0 else 1
    stride_indices_col = indices.stride(0) if indices.dim() > 0 else 1
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(rows, block),)
    
    _min_row_kernel[grid](
        input, output, indices,
        rows, cols,
        stride_input_row, stride_input_col,
        stride_output_row, stride_output_col,
        stride_indices_row, stride_indices_col,
        keepdim,
        BLOCK=block
    )
    
    return (output, indices)
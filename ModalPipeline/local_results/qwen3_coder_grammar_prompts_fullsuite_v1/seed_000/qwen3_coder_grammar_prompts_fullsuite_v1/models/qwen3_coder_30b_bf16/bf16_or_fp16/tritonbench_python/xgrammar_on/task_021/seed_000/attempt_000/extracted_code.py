import torch
import triton
import triton.language as tl

def _max_kernel(input_ptr, output_ptr, indices_ptr, n_elements, dim_size, stride, keepdim, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate the number of rows (elements along the reduced dimension)
    rows = n_elements // dim_size
    
    # Each block handles one row
    row_id = pid
    if row_id >= rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_id * dim_size
    
    # Initialize max value and index
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int64)
    
    # Iterate through all elements in this row
    for i in range(0, dim_size, BLOCK):
        # Calculate offsets
        offsets = row_start + i + tl.arange(0, BLOCK)
        
        # Create mask for valid elements
        mask = offsets < (row_start + dim_size)
        
        # Load input values
        input_vals = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Find max in this block
        block_max = tl.max(input_vals)
        
        # Find index of max in this block
        block_max_idx = tl.argmax(input_vals, 0)
        
        # Update global max
        if block_max > max_val:
            max_val = block_max
            max_idx = block_max_idx + i
    
    # Store results
    if keepdim:
        output_offset = row_id
        indices_offset = row_id
    else:
        output_offset = row_id
        indices_offset = row_id
    
    tl.store(output_ptr + output_offset, max_val)
    tl.store(indices_ptr + indices_offset, max_idx)


def max(input, dim, keepdim=False, *, out=None):
    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim
    
    # Validate dimension
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"Dimension {dim} out of range for tensor with {input.dim()} dimensions")
    
    # Calculate output shape
    output_shape = list(input.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    
    # Create output tensors
    if out is not None:
        max_vals, max_indices = out
        if max_vals.shape != tuple(output_shape):
            raise ValueError("Output tensor max_vals has incorrect shape")
        if max_indices.shape != tuple(output_shape):
            raise ValueError("Output tensor max_indices has incorrect shape")
    else:
        max_vals = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        max_indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Handle scalar case
    if input.numel() == 1:
        max_vals = input.clone()
        max_indices = torch.tensor(0, dtype=torch.long, device=input.device)
        return (max_vals, max_indices)
    
    # For 1D tensor
    if input.dim() == 1:
        max_val = input.max()
        max_idx = input.argmax()
        max_vals = max_val
        max_indices = max_idx
        return (max_vals, max_indices)
    
    # For multi-dimensional tensor
    n_elements = input.numel()
    dim_size = input.shape[dim]
    
    # Calculate stride for the dimension we're reducing
    stride = 1
    for i in range(dim + 1, input.dim()):
        stride *= input.shape[i]
    
    # Launch kernel
    block = 256
    grid = triton.cdiv(n_elements, dim_size)
    
    # Ensure we don't exceed the number of rows
    rows = n_elements // dim_size
    grid = (min(grid, rows),)
    
    _max_kernel[grid](input, max_vals, max_indices, n_elements, dim_size, stride, keepdim, BLOCK=block)
    
    return (max_vals, max_indices)
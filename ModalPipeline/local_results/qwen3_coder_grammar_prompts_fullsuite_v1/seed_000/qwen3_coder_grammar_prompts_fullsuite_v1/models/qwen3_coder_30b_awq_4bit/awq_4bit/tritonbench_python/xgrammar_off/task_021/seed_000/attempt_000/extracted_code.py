import torch
import triton
import triton.language as tl

@triton.jit
def _max_kernel(x_ptr, out_ptr, indices_ptr, n_rows, n_cols, dim_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    
    # Load the row data
    row_offsets = pid * n_cols
    row_ptr = x_ptr + row_offsets
    
    # Initialize max and index
    max_val = tl.full([], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([], 0, dtype=tl.int32)
    
    # Iterate through the dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load values
        values = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Find max in this block
        block_max = tl.max(values)
        block_max_idx = tl.arg_max(values)
        
        # Update global max
        if block_max > max_val:
            max_val = block_max
            max_idx = block_max_idx + i
    
    # Store results
    tl.store(out_ptr + pid, max_val)
    tl.store(indices_ptr + pid, max_idx)

def max(input, dim, keepdim=False, *, out=None):
    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim
    
    # Validate dimension
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"Dimension {dim} is out of range for input with {input.dim()} dimensions")
    
    # Get output shape
    input_shape = input.shape
    output_shape = list(input_shape)
    
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    
    # Create output tensors
    if out is not None:
        max_values = out[0]
        max_indices = out[1]
    else:
        max_values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        max_indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Get dimensions
    n_rows = 1
    n_cols = 1
    dim_size = input_shape[dim]
    
    # Calculate total elements
    for i in range(input.dim()):
        if i == dim:
            n_cols *= input_shape[i]
        else:
            n_rows *= input_shape[i]
    
    # Handle special case where we're reducing all dimensions
    if input.dim() == 1:
        # For 1D tensor, just find max and argmax
        max_val = input.max()
        max_idx = input.argmax()
        
        if out is not None:
            out[0].copy_(max_val)
            out[1].copy_(max_idx)
        else:
            return (max_val, max_idx)
    
    # For multi-dimensional case, use Triton kernel
    if n_rows > 0:
        block_size = 256
        grid_size = triton.cdiv(n_rows, block_size)
        
        # Create a temporary contiguous tensor for kernel
        if input.is_contiguous():
            x_ptr = input.data_ptr()
        else:
            input = input.contiguous()
            x_ptr = input.data_ptr()
        
        # Launch kernel
        _max_kernel[grid_size](
            x_ptr,
            max_values.data_ptr(),
            max_indices.data_ptr(),
            n_rows,
            n_cols,
            dim_size,
            BLOCK_SIZE=block_size
        )
    
    if out is not None:
        return out
    else:
        return (max_values, max_indices)

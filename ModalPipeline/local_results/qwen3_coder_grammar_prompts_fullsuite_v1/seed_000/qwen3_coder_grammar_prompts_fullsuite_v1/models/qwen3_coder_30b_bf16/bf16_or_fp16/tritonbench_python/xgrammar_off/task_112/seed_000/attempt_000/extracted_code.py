import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    input_ptr, 
    output_ptr, 
    indices_ptr,
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Initialize with first element
    min_val = tl.load(input_ptr + row_id * n_cols)
    min_idx = 0
    
    # Iterate through remaining elements
    for col in range(1, n_cols):
        val = tl.load(input_ptr + row_id * n_cols + col)
        if val < min_val:
            min_val = val
            min_idx = col
    
    # Store results
    tl.store(output_ptr + row_id, min_val)
    tl.store(indices_ptr + row_id, min_idx)

@triton.jit
def min_kernel_with_indices(
    input_ptr, 
    output_ptr, 
    indices_ptr,
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Initialize with first element
    min_val = tl.load(input_ptr + row_id * n_cols)
    min_idx = 0
    
    # Iterate through remaining elements
    for col in range(1, n_cols):
        val = tl.load(input_ptr + row_id * n_cols + col)
        if val < min_val:
            min_val = val
            min_idx = col
    
    # Store results
    tl.store(output_ptr + row_id, min_val)
    tl.store(indices_ptr + row_id, min_idx)

def min(input, dim, keepdim=False, *, out=None):
    if dim < 0:
        dim = input.dim() + dim
    
    if dim >= input.dim():
        raise ValueError("dim out of range")
    
    # Get input shape
    shape = input.shape
    n_rows = 1
    n_cols = shape[dim]
    
    # Calculate number of rows (product of all dimensions except dim)
    for i, s in enumerate(shape):
        if i != dim:
            n_rows *= s
    
    # Create output tensors
    if out is not None:
        output, indices = out
    else:
        output = torch.empty(n_rows, dtype=input.dtype, device=input.device)
        indices = torch.empty(n_rows, dtype=torch.long, device=input.device)
    
    # Prepare input for kernel (flatten all dimensions except dim)
    if dim == 0:
        input_flat = input.view(n_rows, n_cols)
    else:
        # Transpose to make dim the last dimension for easier processing
        perm = list(range(input.dim()))
        perm[0], perm[dim] = perm[dim], perm[0]
        input_transposed = input.permute(perm)
        input_flat = input_transposed.view(n_rows, n_cols)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1, 1)
    
    min_kernel[grid](
        input_flat,
        output,
        indices,
        n_rows,
        n_cols,
        BLOCK_SIZE
    )
    
    # Reshape output if keepdim is True
    if keepdim:
        new_shape = list(shape)
        new_shape[dim] = 1
        output = output.view(new_shape)
        indices = indices.view(new_shape)
    else:
        # Remove the dim dimension from output shape
        new_shape = list(shape)
        new_shape.pop(dim)
        output = output.view(new_shape)
        indices = indices.view(new_shape)
    
    return output, indices

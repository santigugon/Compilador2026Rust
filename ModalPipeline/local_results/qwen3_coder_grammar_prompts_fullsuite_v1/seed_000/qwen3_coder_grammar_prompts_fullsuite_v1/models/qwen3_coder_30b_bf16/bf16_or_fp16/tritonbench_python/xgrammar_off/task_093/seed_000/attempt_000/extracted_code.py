import torch
import triton
import triton.language as tl

@triton.jit
def softmax_log_kernel(
    input_ptr, 
    output_ptr, 
    n_cols, 
    n_rows, 
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    if dim == -1 or dim == 1:
        # Compute along the last dimension
        input_row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=col_offsets < n_cols)
        # Apply log
        input_row = tl.log(input_row)
        # Compute max for numerical stability
        row_max = tl.max(input_row, axis=0)
        # Subtract max for numerical stability
        input_row = input_row - row_max
        # Compute exp
        input_row = tl.exp(input_row)
        # Compute sum
        row_sum = tl.sum(input_row, axis=0)
        # Normalize
        output_row = input_row / row_sum
    else:
        # Compute along other dimensions (simplified for 2D case)
        # This is a simplified version assuming dim=0 for 2D case
        # For more general cases, we'd need to handle transposition
        col_idx = tl.program_id(1)
        if col_idx < n_cols:
            # Load column data
            col_data = tl.load(input_ptr + col_idx + col_offsets * n_cols, mask=col_offsets < n_rows)
            # Apply log
            col_data = tl.log(col_data)
            # Compute max for numerical stability
            col_max = tl.max(col_data, axis=0)
            # Subtract max for numerical stability
            col_data = col_data - col_max
            # Compute exp
            col_data = tl.exp(col_data)
            # Compute sum
            col_sum = tl.sum(col_data, axis=0)
            # Normalize
            output_col = col_data / col_sum
            # Store result
            tl.store(output_ptr + col_idx + col_offsets * n_cols, output_col, mask=col_offsets < n_rows)
        return
    
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output_row, mask=col_offsets < n_cols)

def softmax_log(input, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    if dim < 0:
        dim = input.dim() + dim
    
    if dim == 0:
        # For dim=0, we need to transpose and compute along dim=1
        input = input.transpose(0, 1)
        dim = 1
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Get dimensions
    n_rows, n_cols = input.shape
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1)
    
    softmax_log_kernel[grid](
        input,
        output,
        n_cols,
        n_rows,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if dim == 1 and input.dim() == 2:
        # Transpose back if needed
        output = output.transpose(0, 1)
    
    return output

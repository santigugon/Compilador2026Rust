import torch
import triton
import triton.language as tl

@triton.jit
def softmax_log_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    row_start = row_id * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    input_ptrs = input_ptr + row_start + col_offsets
    input_vals = tl.load(input_ptrs, mask=col_offsets < n_cols)
    
    # Apply log
    input_vals = tl.log(input_vals)
    
    # Compute max for numerical stability
    if dim == -1 or dim == 1:
        max_val = tl.max(input_vals, axis=0)
        input_vals = input_vals - max_val
        exp_vals = tl.exp(input_vals)
        sum_exp = tl.sum(exp_vals, axis=0)
        output_vals = exp_vals / sum_exp
    else:
        # For other dimensions, we need to handle differently
        # This is a simplified version for 2D tensors
        max_val = tl.max(input_vals, axis=0)
        input_vals = input_vals - max_val
        exp_vals = tl.exp(input_vals)
        sum_exp = tl.sum(exp_vals, axis=0)
        output_vals = exp_vals / sum_exp
    
    # Store output
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, output_vals, mask=col_offsets < n_cols)

def softmax_log(input, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    shape = input.shape
    n_rows = shape[0]
    n_cols = shape[1] if len(shape) > 1 else 1
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1, 1)
    
    softmax_log_kernel[grid](
        input,
        output,
        n_rows,
        n_cols,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

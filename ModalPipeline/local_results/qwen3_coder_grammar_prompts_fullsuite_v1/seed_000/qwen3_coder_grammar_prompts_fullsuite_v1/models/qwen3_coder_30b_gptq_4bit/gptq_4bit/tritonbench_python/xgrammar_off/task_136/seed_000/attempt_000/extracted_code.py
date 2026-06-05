import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    output_row = output_ptr + row * n_cols
    input_row = input_ptr + row * n_cols
    
    # Load input data
    values = tl.load(input_row + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Compute softmax
    max_val = tl.max(values, axis=0)
    exp_vals = tl.exp(values - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    tl.store(output_row + tl.arange(0, BLOCK_SIZE), softmax_vals, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get output tensor
    output = torch.empty_like(input)
    
    # Get dimensions
    shape = input.shape
    n_rows = 1
    n_cols = shape[dim]
    
    # Compute total number of rows
    for i in range(len(shape)):
        if i != dim:
            n_rows *= shape[i]
    
    # Set block size
    BLOCK_SIZE = 1024
    if n_cols > BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Launch kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        output,
        input,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

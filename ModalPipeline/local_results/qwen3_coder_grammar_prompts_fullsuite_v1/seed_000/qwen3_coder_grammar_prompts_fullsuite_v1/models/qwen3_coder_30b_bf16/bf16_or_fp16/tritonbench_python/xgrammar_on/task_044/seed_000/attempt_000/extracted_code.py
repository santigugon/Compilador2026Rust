import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr,  # pointer to input tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor
    output_ptr,  # pointer to output tensor
    input_row_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    n_cols,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row = tl.program_id(0)
    
    # Compute the input and weight pointers for this row
    input_row_ptr = input_ptr + row * input_row_stride
    weight_row_ptr = weight_ptr + row * weight_row_stride
    output_row_ptr = output_ptr + row * output_row_stride
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute linear transformation
    for col in range(0, n_cols, BLOCK_SIZE):
        # Load input and weight
        input_vals = tl.load(input_row_ptr + col * tl.arange(0, BLOCK_SIZE))
        weight_vals = tl.load(weight_row_ptr + col * tl.arange(0, BLOCK_SIZE))
        
        # Accumulate
        accumulator += input_vals * weight_vals
    
    # Add bias if present
    if bias_ptr is not None:
        bias_row_ptr = bias_ptr + row * weight_row_stride
        bias_vals = tl.load(bias_row_ptr)
        accumulator += bias_vals
    
    # Apply softplus activation
    # Softplus = log(1 + exp(beta * x))
    # For numerical stability, if x > threshold, use x instead of softplus
    softplus_vals = tl.where(
        accumulator > threshold,
        accumulator,
        tl.log(1.0 + tl.exp(beta * accumulator))
    )
    
    # Store result
    tl.store(output_row_ptr, softplus_vals)


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    # Ensure input is contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    n_rows, n_cols = input.shape
    n_out = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(n_rows, n_out, device=input.device, dtype=input.dtype)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (n_rows,)
    
    # Determine if bias is present
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    softplus_linear_kernel[grid](
        input.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        input.stride(0),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        n_cols,
        beta,
        threshold,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
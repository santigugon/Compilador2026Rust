import torch
import triton
import triton.language as tl

@triton.jit
def _elu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows: tl.constexpr, n_cols: tl.constexpr, n_features: tl.constexpr,
    alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return
    
    # Load input row
    input_row = tl.load(input_ptr + row * n_features, mask=tl.arange(0, n_features) < n_features)
    
    # Compute linear transformation: input @ weight.T + bias
    output_row = tl.zeros((n_cols,), dtype=tl.float32)
    for i in range(0, n_features, BLOCK_SIZE):
        # Load weight column slice
        weight_offsets = tl.arange(i, i + BLOCK_SIZE)
        weight_mask = weight_offsets < n_features
        weight_col = tl.load(weight_ptr + tl.arange(0, n_cols)[:, None] * n_features + weight_offsets[None, :], mask=weight_mask[None, :], other=0.0)
        
        # Load input slice
        input_slice = tl.load(input_row + weight_offsets, mask=weight_mask, other=0.0)
        
        # Compute dot product
        output_row += tl.sum(input_slice[None, :] * weight_col, axis=1)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, n_cols), mask=tl.arange(0, n_cols) < n_cols)
        output_row += bias
    
    # Apply ELU activation
    output_row = tl.where(output_row > 0, output_row, alpha * tl.exp(output_row) - alpha)
    
    # Store result
    tl.store(output_ptr + row * n_cols, output_row, mask=tl.arange(0, n_cols) < n_cols)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Validate inputs
    if input.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional")
    if weight.dim() != 2:
        raise ValueError("Weight tensor must be 2-dimensional")
    if input.size(1) != weight.size(1):
        raise ValueError("Input size and weight size mismatch")
    if bias is not None and bias.size(0) != weight.size(0):
        raise ValueError("Bias size mismatch with weight size")
    
    # Prepare output tensor
    n_rows = input.size(0)
    n_cols = weight.size(0)
    
    # Create output tensor
    out = torch.empty(n_rows, n_cols, dtype=input.dtype, device=input.device)
    
    # Handle inplace operation
    if inplace:
        # For inplace, we need to ensure the output tensor is the same as input
        # But since we're doing a linear transformation, we can't do true inplace
        # We'll just use the same tensor for output
        out = torch.empty_like(input)
        # This is a simplified approach - in practice, inplace would require
        # a different approach to handle the transformation properly
        # For now, we'll proceed with the standard approach
        pass
    
    # Launch kernel
    block_size = 256
    grid_size = triton.cdiv(n_rows, 1)
    
    # Create a temporary tensor for the linear transformation result
    temp_out = torch.empty(n_rows, n_cols, dtype=input.dtype, device=input.device)
    
    # Launch the kernel
    _elu_linear_kernel[grid_size](
        input, weight, bias, temp_out,
        n_rows, n_cols, input.size(1),
        alpha, BLOCK_SIZE=block_size
    )
    
    # Return the result
    return temp_out

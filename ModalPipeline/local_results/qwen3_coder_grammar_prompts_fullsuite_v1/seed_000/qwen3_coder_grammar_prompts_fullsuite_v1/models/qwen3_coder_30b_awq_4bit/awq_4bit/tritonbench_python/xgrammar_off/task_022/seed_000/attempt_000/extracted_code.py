import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, out_features,
    BLOCK_SIZE: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Load input row
    input_row = tl.load(input_ptr + row_idx * input_row_stride, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Compute linear transformation: output = input @ weight.T + bias
    output_row = tl.zeros((out_features,), dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        col_mask = (tl.arange(0, BLOCK_SIZE) + i) < n_cols
        weight_col = tl.load(weight_ptr + i * weight_col_stride, mask=col_mask)
        
        # Compute dot product
        for j in range(out_features):
            weight_val = tl.load(weight_ptr + j * weight_row_stride + i * weight_col_stride, mask=col_mask)
            input_val = tl.load(input_row + i, mask=col_mask)
            output_row[j] += input_val * weight_val
    
    # Add bias if present
    if USE_BIAS:
        bias_row = tl.load(bias_ptr, mask=tl.arange(0, out_features) < out_features)
        output_row += bias_row
    
    # Apply log_softmax
    max_val = tl.max(output_row, axis=0)
    exp_row = tl.exp(output_row - max_val)
    sum_exp = tl.sum(exp_row, axis=0)
    log_softmax_row = tl.log(exp_row / sum_exp) + max_val
    
    # Store result
    tl.store(output_ptr + row_idx * output_row_stride, log_softmax_row, mask=tl.arange(0, out_features) < out_features)

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is 2D for processing
    original_shape = input.shape
    if len(original_shape) > 2:
        input = input.view(-1, original_shape[-1])
    
    # Get dimensions
    batch_size = input.shape[0]
    in_features = input.shape[1]
    out_features = weight.shape[0]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_features, dtype=torch.float32, device=input.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    grid = (num_blocks, 1, 1)
    use_bias = bias is not None
    
    log_softmax_linear_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if use_bias else 0,
        output_ptr=output.data_ptr(),
        input_row_stride=input.stride(0),
        weight_row_stride=weight.stride(0),
        weight_col_stride=weight.stride(1),
        output_row_stride=output.stride(0),
        output_col_stride=output.stride(1),
        n_rows=batch_size,
        n_cols=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BIAS=use_bias
    )
    
    # Reshape output to original shape
    if len(original_shape) > 2:
        output = output.view(*original_shape[:-1], out_features)
    
    return output

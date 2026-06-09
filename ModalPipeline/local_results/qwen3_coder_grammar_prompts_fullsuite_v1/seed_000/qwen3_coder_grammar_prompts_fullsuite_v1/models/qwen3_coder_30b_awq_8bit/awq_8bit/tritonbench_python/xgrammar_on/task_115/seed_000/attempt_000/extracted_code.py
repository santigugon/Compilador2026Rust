import torch
import triton
import triton.language as tl

@triton.jit
def _tanh_linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, n_rows, n_cols, weight_stride_row, weight_stride_col, bias_stride, output_stride_row, output_stride_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return
    
    # Load input row
    input_row = tl.load(input_ptr + row * n_cols, mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Compute linear transformation
    output_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        # Load weight column
        weight_col = tl.load(weight_ptr + i + tl.arange(0, BLOCK_SIZE) * weight_stride_col, mask=tl.arange(0, BLOCK_SIZE) + i < n_cols, other=0.0)
        # Compute dot product
        output_row += tl.sum(input_row * weight_col)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_row = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE) * bias_stride, mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
        output_row += bias_row
    
    # Apply tanh
    output_row = 2.0 / (1.0 + tl.exp(-2.0 * output_row)) - 1.0
    
    # Store result
    tl.store(output_ptr + row * output_stride_row + tl.arange(0, BLOCK_SIZE) * output_stride_col, output_row, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def tanh_linear(input, weight, bias=None):
    # Validate input dimensions
    assert input.dim() >= 2, "input must have at least 2 dimensions"
    assert weight.dim() == 2, "weight must be 2-dimensional"
    assert weight.size(1) == input.size(-1), "weight size(1) must match input size(-1)"
    
    if bias is not None:
        assert bias.dim() == 1, "bias must be 1-dimensional"
        assert bias.size(0) == weight.size(0), "bias size(0) must match weight size(0)"
    
    # Flatten input to 2D
    input_shape = input.shape
    input_2d = input.view(-1, input.size(-1))
    
    # Prepare output
    out_features = weight.size(0)
    output = torch.empty(input_2d.size(0), out_features, dtype=input.dtype, device=input.device)
    
    # Get dimensions
    n_rows = input_2d.size(0)
    n_cols = input_2d.size(1)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_rows, 1),)
    
    # Prepare pointers
    input_ptr = input_2d.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    # Compute strides
    weight_stride_row = weight.stride(0)
    weight_stride_col = weight.stride(1)
    bias_stride = bias.stride(0) if bias is not None else 0
    output_stride_row = output.stride(0)
    output_stride_col = output.stride(1)
    
    _tanh_linear_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        n_rows, n_cols,
        weight_stride_row, weight_stride_col,
        bias_stride,
        output_stride_row, output_stride_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to match input shape
    output = output.view(input_shape[:-1] + (out_features,))
    
    return output
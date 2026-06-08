import torch
import triton
import triton.language as tl

@triton.jit
def _linear_relu_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, out_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    
    # Load input row
    input_offsets = pid * input_row_stride + tl.arange(0, n_cols)
    input_row = tl.load(input_ptr + input_offsets, mask=input_offsets < n_rows * input_row_stride, other=0.0)
    
    # Linear transformation
    output_offsets = pid * output_row_stride + tl.arange(0, out_features)
    output_row = tl.zeros((out_features,), dtype=tl.float32)
    
    for i in range(n_cols):
        weight_offsets = i * weight_col_stride + tl.arange(0, out_features)
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_offsets < weight_row_stride * weight_col_stride, other=0.0)
        output_row += input_row[i] * weight_vals
    
    # Add bias
    if bias_ptr is not None:
        bias_offsets = tl.arange(0, out_features)
        bias_vals = tl.load(bias_ptr + bias_offsets, mask=bias_offsets < out_features, other=0.0)
        output_row += bias_vals
    
    # Apply ReLU
    output_row = tl.where(output_row > 0, output_row, 0.0)
    
    # Layer normalization
    # Compute mean and variance
    mean = tl.sum(output_row) / out_features
    var = tl.sum((output_row - mean) * (output_row - mean)) / out_features
    
    # Normalize
    normalized = (output_row - mean) / tl.sqrt(var + eps)
    
    # Store result
    tl.store(output_ptr + output_offsets, normalized, mask=output_offsets < n_rows * output_row_stride)

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Handle normalized_shape
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Validate input dimensions
    assert input.shape[-1] == weight.shape[1], "Input feature dimension must match weight dimension"
    assert weight.shape[0] == len(normalized_shape), "Output features must match normalized shape"
    
    # Handle bias
    if bias is not None:
        assert bias.shape[0] == weight.shape[0], "Bias shape must match weight output dimension"
    else:
        bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    
    # Prepare output tensor
    out = torch.empty(input.shape[:-1] + (weight.shape[0],), device=input.device, dtype=input.dtype)
    
    # Get dimensions
    n_rows = input.numel() // input.shape[-1]
    n_cols = input.shape[-1]
    out_features = weight.shape[0]
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    
    # Launch kernel
    _linear_relu_norm_kernel[grid](
        input, weight, bias, out,
        input.stride(0), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        n_rows, n_cols, out_features,
        eps,
        BLOCK_SIZE
    )
    
    return out

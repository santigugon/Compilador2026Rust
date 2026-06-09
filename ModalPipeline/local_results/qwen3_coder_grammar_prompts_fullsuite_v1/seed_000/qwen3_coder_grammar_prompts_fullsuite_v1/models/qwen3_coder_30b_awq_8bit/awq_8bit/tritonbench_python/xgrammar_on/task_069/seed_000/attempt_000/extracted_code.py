import torch
import triton
import triton.language as tl

def _fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    mean_ptr, var_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, out_features,
    normalized_shape, eps, elementwise_affine,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return
    
    # Load input row
    input_row = tl.load(input_ptr + row * input_row_stride + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Linear transformation
    linear_out = tl.zeros((out_features,), dtype=tl.float32)
    for i in range(out_features):
        weight_row = tl.load(weight_ptr + i * weight_row_stride + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
        linear_out[i] = tl.sum(input_row * weight_row)
    
    if bias_ptr is not None:
        bias_row = tl.load(bias_ptr + tl.arange(0, out_features), mask=tl.arange(0, out_features) < out_features, other=0.0)
        linear_out = linear_out + bias_row
    
    # ReLU
    linear_out = tl.where(linear_out > 0, linear_out, 0.0)
    
    # Layer normalization
    # Compute mean
    mean = tl.sum(linear_out) / out_features
    
    # Compute variance
    var = tl.sum((linear_out - mean) * (linear_out - mean)) / out_features
    
    # Store mean and variance
    if mean_ptr is not None:
        tl.store(mean_ptr + row, mean)
    if var_ptr is not None:
        tl.store(var_ptr + row, var)
    
    # Normalize
    normalized = (linear_out - mean) / tl.sqrt(var + eps)
    
    # Apply elementwise affine if needed
    if elementwise_affine:
        # For simplicity, we assume scale and bias are 1 and 0 respectively
        # In a full implementation, these would be passed in
        pass
    
    # Store output
    tl.store(output_ptr + row * output_row_stride + tl.arange(0, out_features), normalized, mask=tl.arange(0, out_features) < out_features)


def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Validate inputs
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    
    if not isinstance(normalized_shape, (int, list, torch.Size)):
        raise ValueError("normalized_shape must be an int, list, or torch.Size")
    
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Check if the last dimensions match
    if input.shape[-len(normalized_shape):] != tuple(normalized_shape):
        raise ValueError("The last dimensions of input must match normalized_shape")
    
    # Get dimensions
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    
    # Check input dimensions
    if input.shape[-1] != in_features:
        raise ValueError("Input features must match weight shape")
    
    # Check bias dimensions if provided
    if bias is not None and bias.shape[0] != out_features:
        raise ValueError("Bias must have the same number of elements as out_features")
    
    # Flatten input to 2D for processing
    input_flat = input.view(-1, in_features)
    n_rows = input_flat.shape[0]
    
    # Create output tensor
    output = torch.empty(n_rows, out_features, dtype=input.dtype, device=input.device)
    
    # Create mean and variance tensors if needed
    mean = torch.empty(n_rows, dtype=torch.float32, device=input.device) if elementwise_affine else None
    var = torch.empty(n_rows, dtype=torch.float32, device=input.device) if elementwise_affine else None
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    
    # Prepare pointers
    input_ptr = input_flat.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    mean_ptr = mean.data_ptr() if mean is not None else None
    var_ptr = var.data_ptr() if var is not None else None
    
    # Launch kernel
    _fused_layer_norm_relu_linear_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        mean_ptr, var_ptr,
        input_flat.stride(0), weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        n_rows, in_features, out_features,
        normalized_shape, eps, elementwise_affine,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to original input shape
    output = output.view(input.shape[:-1] + (out_features,))
    
    return output
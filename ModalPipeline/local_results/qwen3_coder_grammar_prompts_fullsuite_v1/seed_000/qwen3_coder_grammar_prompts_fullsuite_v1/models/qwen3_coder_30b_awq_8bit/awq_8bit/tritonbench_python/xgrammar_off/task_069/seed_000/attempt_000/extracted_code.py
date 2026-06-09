import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
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
    
    # Linear transformation: input @ weight.T + bias
    linear_out = tl.zeros((out_features,), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (tl.arange(0, BLOCK_SIZE) + i) < n_cols
        weight_col = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE) + i * weight_col_stride, mask=mask, other=0.0)
        linear_out += tl.sum(input_row[None, :] * weight_col[:, None], axis=0)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, out_features), mask=tl.arange(0, out_features) < out_features, other=0.0)
        linear_out += bias
    
    # Apply ReLU
    linear_out = tl.where(linear_out > 0, linear_out, 0.0)
    
    # Layer normalization
    if elementwise_affine:
        # Compute mean and variance
        mean = tl.sum(linear_out) / normalized_shape
        var = tl.sum((linear_out - mean) * (linear_out - mean)) / normalized_shape
        
        # Normalize
        normalized = (linear_out - mean) / tl.sqrt(var + eps)
        
        # Apply affine transformation (scale and bias)
        scale = tl.load(weight_ptr + tl.arange(0, out_features) + n_cols * weight_col_stride, mask=tl.arange(0, out_features) < out_features, other=1.0)
        bias_affine = tl.load(bias_ptr + tl.arange(0, out_features) + n_cols, mask=tl.arange(0, out_features) < out_features, other=0.0)
        normalized = normalized * scale + bias_affine
    else:
        # Compute mean and variance
        mean = tl.sum(linear_out) / normalized_shape
        var = tl.sum((linear_out - mean) * (linear_out - mean)) / normalized_shape
        
        # Normalize
        normalized = (linear_out - mean) / tl.sqrt(var + eps)
    
    # Store result
    tl.store(output_ptr + row * output_row_stride + tl.arange(0, out_features), normalized, mask=tl.arange(0, out_features) < out_features)

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Validate inputs
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    
    if not isinstance(normalized_shape, (int, list, torch.Size)):
        raise ValueError("normalized_shape must be an int, list, or torch.Size")
    
    if isinstance(normalized_shape, (list, torch.Size)):
        normalized_shape = int(torch.Size(normalized_shape).numel())
    
    # Ensure input is 2D for processing
    input_shape = input.shape
    if len(input_shape) < 2:
        input = input.unsqueeze(0)
    
    n_rows, n_cols = input.shape[0], input.shape[1]
    out_features = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(n_rows, out_features, dtype=input.dtype, device=input.device)
    
    # Prepare pointers and strides
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    input_row_stride = input.stride(0)
    weight_row_stride = weight.stride(0)
    weight_col_stride = weight.stride(1)
    output_row_stride = output.stride(0)
    output_col_stride = output.stride(1)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_rows, 1),)
    
    # Adjust for the case where we have a bias tensor
    if bias is not None:
        # We need to make sure we have enough space for the bias in the kernel
        # This is a simplified approach - in practice, we'd need to handle this more carefully
        pass
    
    _fused_layer_norm_relu_linear_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        input_row_stride, weight_row_stride, weight_col_stride,
        output_row_stride, output_col_stride,
        n_rows, n_cols, out_features,
        normalized_shape, eps, elementwise_affine,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output if needed
    if len(input_shape) < 2:
        output = output.squeeze(0)
    
    return output

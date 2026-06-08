import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_features: tl.constexpr, n_out: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    
    # Load input row
    input_offsets = row * input_row_stride + tl.arange(0, BLOCK)
    input_row = tl.load(input_ptr + input_offsets, mask=input_offsets < n_features, other=0.0)
    
    # Linear transformation: input @ weight.T + bias
    linear_out = tl.zeros((n_out,), dtype=tl.float32)
    for i in range(0, n_features, BLOCK):
        input_offsets = row * input_row_stride + i + tl.arange(0, BLOCK)
        input_vals = tl.load(input_ptr + input_offsets, mask=input_offsets < n_features, other=0.0)
        
        for j in range(n_out):
            weight_offsets = j * weight_row_stride + i + tl.arange(0, BLOCK)
            weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_offsets < n_features, other=0.0)
            linear_out[j] += tl.sum(input_vals * weight_vals)
    
    # Add bias
    if bias_ptr is not None:
        bias_offsets = tl.arange(0, n_out)
        bias_vals = tl.load(bias_ptr + bias_offsets, mask=bias_offsets < n_out, other=0.0)
        linear_out += bias_vals
    
    # Apply ReLU
    linear_out = tl.where(linear_out > 0.0, linear_out, 0.0)
    
    # Layer normalization
    # Compute mean
    mean = tl.sum(linear_out) / n_out
    
    # Compute variance
    diff = linear_out - mean
    variance = tl.sum(diff * diff) / n_out
    
    # Normalize
    std = tl.sqrt(variance + eps)
    normalized = diff / std
    
    # Store result
    output_offsets = row * output_row_stride + tl.arange(0, n_out)
    tl.store(output_ptr + output_offsets, normalized, mask=output_offsets < n_out)

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Handle normalized_shape
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Validate shapes
    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    assert weight.dim() == 2, "Weight must be 2D"
    assert weight.shape[1] == input.shape[-1], "Weight input dimension must match input last dimension"
    assert weight.shape[0] == normalized_shape[0], "Weight output dimension must match normalized_shape"
    
    if bias is not None:
        assert bias.shape[0] == weight.shape[0], "Bias must have same size as weight output dimension"
    
    # Prepare output tensor
    input_shape = input.shape
    output_shape = input_shape[:-1] + (weight.shape[0],)
    out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Flatten input for processing
    input_flat = input.view(-1, input.shape[-1])
    out_flat = out.view(-1, weight.shape[0])
    
    # Get strides
    input_row_stride = input_flat.stride(0)
    weight_row_stride = weight.stride(0)
    weight_col_stride = weight.stride(1)
    output_row_stride = out_flat.stride(0)
    output_col_stride = out_flat.stride(1)
    
    # Launch kernel
    n_rows = input_flat.shape[0]
    n_features = input_flat.shape[1]
    n_out = weight.shape[0]
    block = 256
    grid = (n_rows,)
    
    # Create a temporary tensor for the linear output before normalization
    temp_linear = torch.empty(n_rows, n_out, dtype=torch.float32, device=input.device)
    
    # First compute linear transformation
    _linear_kernel[grid](
        input_flat, weight, bias, temp_linear,
        input_row_stride, weight_row_stride, weight_col_stride,
        temp_linear.stride(0), temp_linear.stride(1),
        n_features, n_out, BLOCK=block
    )
    
    # Then apply ReLU and LayerNorm
    _relu_layer_norm_kernel[grid](
        temp_linear, out_flat,
        temp_linear.stride(0), out_flat.stride(0),
        n_out, eps, BLOCK=block
    )
    
    return out

@triton.jit
def _linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_features: tl.constexpr, n_out: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    
    # Linear transformation: input @ weight.T + bias
    for j in range(n_out):
        linear_out = 0.0
        for i in range(0, n_features, BLOCK):
            input_offsets = row * input_row_stride + i + tl.arange(0, BLOCK)
            input_vals = tl.load(input_ptr + input_offsets, mask=input_offsets < n_features, other=0.0)
            
            weight_offsets = j * weight_row_stride + i + tl.arange(0, BLOCK)
            weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_offsets < n_features, other=0.0)
            
            linear_out += tl.sum(input_vals * weight_vals)
        
        # Add bias if provided
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + j, mask=j < n_out, other=0.0)
            linear_out += bias_val
        
        # Store result
        output_offsets = row * output_row_stride + j
        tl.store(output_ptr + output_offsets, linear_out, mask=output_offsets < n_out)

@triton.jit
def _relu_layer_norm_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_out: tl.constexpr, eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    
    # Load input row
    input_offsets = row * input_row_stride + tl.arange(0, BLOCK)
    input_row = tl.load(input_ptr + input_offsets, mask=input_offsets < n_out, other=0.0)
    
    # Apply ReLU
    input_row = tl.where(input_row > 0.0, input_row, 0.0)
    
    # Layer normalization
    # Compute mean
    mean = tl.sum(input_row) / n_out
    
    # Compute variance
    diff = input_row - mean
    variance = tl.sum(diff * diff) / n_out
    
    # Normalize
    std = tl.sqrt(variance + eps)
    normalized = diff / std
    
    # Store result
    output_offsets = row * output_row_stride + tl.arange(0, n_out)
    tl.store(output_ptr + output_offsets, normalized, mask=output_offsets < n_out)

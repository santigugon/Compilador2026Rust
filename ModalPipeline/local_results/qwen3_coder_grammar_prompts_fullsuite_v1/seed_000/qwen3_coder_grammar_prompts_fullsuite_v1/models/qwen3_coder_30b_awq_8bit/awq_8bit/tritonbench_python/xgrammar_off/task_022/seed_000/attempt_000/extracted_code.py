import torch
import triton
import triton.language as tl

@triton.jit
def _log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    input_size: tl.constexpr,
    weight_size: tl.constexpr,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Load input row
    input_offsets = row_idx * input_row_stride + tl.arange(0, BLOCK_SIZE)
    input_block = tl.load(input_ptr + input_offsets, mask=input_offsets < input_size, other=0.0)
    
    # Compute linear transformation: input @ weight.T + bias
    output_offsets = row_idx * output_row_stride + tl.arange(0, BLOCK_SIZE)
    output_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute matrix multiplication
    for i in range(0, weight_size, BLOCK_SIZE):
        # Load weight column
        weight_offsets = i + tl.arange(0, BLOCK_SIZE)
        weight_block = tl.load(weight_ptr + weight_offsets * weight_col_stride, mask=weight_offsets < weight_size, other=0.0)
        
        # Compute dot product
        dot_product = tl.sum(input_block * weight_block)
        output_block += dot_product
    
    # Add bias if present
    if HAS_BIAS:
        bias_offsets = tl.arange(0, BLOCK_SIZE)
        bias_block = tl.load(bias_ptr + bias_offsets, mask=bias_offsets < output_size, other=0.0)
        output_block += bias_block
    
    # Apply log_softmax
    # First, find max for numerical stability
    max_val = tl.max(output_block, axis=0)
    exp_block = tl.exp(output_block - max_val)
    sum_exp = tl.sum(exp_block, axis=0)
    log_softmax_block = output_block - max_val - tl.log(sum_exp)
    
    # Store result
    tl.store(output_ptr + output_offsets, log_softmax_block, mask=output_offsets < output_size)

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
    
    # Get dimensions
    input_shape = input.shape
    weight_shape = weight.shape
    batch_dims = input_shape[:-1]
    in_features = input_shape[-1]
    out_features = weight_shape[0]
    
    # Flatten input to 2D for processing
    input_flat = input.view(-1, in_features)
    batch_size = input_flat.shape[0]
    
    # Create output tensor
    output_shape = batch_dims + (out_features,)
    output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    
    # Prepare strides
    input_row_stride = input_flat.stride(0)
    weight_row_stride = weight.stride(0)
    weight_col_stride = weight.stride(1)
    output_row_stride = output.stride(0)
    output_col_stride = output.stride(-1) if len(output.shape) > 1 else 1
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    # Determine if we have bias
    has_bias = bias is not None
    
    if has_bias:
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = 0
    
    _log_softmax_linear_kernel[grid](
        input_flat.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        input_row_stride,
        weight_row_stride,
        weight_col_stride,
        output_row_stride,
        output_col_stride,
        input_flat.numel(),
        weight_shape[1],
        out_features,
        BLOCK_SIZE,
        has_bias
    )
    
    # Return with proper shape
    return output.view(output_shape)

import torch
import triton
import triton.language as tl

@triton.jit
def tanh_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, out_features,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    
    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride
    
    for i in range(0, out_features, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < out_features
        
        # Compute linear transformation
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, n_cols, BLOCK_SIZE):
            input_cols = j + tl.arange(0, BLOCK_SIZE)
            input_mask = input_cols < n_cols
            
            input_vals = tl.load(input_row + input_cols, mask=input_mask, other=0.0)
            weight_vals = tl.load(weight_ptr + cols * weight_col_stride + j, mask=mask, other=0.0)
            acc += tl.sum(input_vals * weight_vals, axis=0)
        
        # Add bias if present
        if bias_ptr is not None:
            bias_vals = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            acc += bias_vals
        
        # Apply tanh activation
        tanh_vals = tl.tanh(acc)
        tl.store(output_row + cols, tanh_vals, mask=mask)

def tanh_linear(input, weight, bias=None):
    if input.dim() < 2:
        raise ValueError("input tensor must have at least 2 dimensions")
    
    input_shape = input.shape
    batch_size = 1
    for dim in input_shape[:-1]:
        batch_size *= dim
    in_features = input_shape[-1]
    
    out_features = weight.shape[0]
    
    if in_features != weight.shape[1]:
        raise ValueError("input features must match weight matrix dimensions")
    
    if bias is not None and bias.shape[0] != out_features:
        raise ValueError("bias tensor must match output features")
    
    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    
    # Launch kernel
    grid = (batch_size, 1, 1)
    BLOCK_SIZE = 128
    
    if bias is not None:
        tanh_linear_kernel[grid](
            input, weight, bias, output,
            input.stride(-2), weight.stride(-2), weight.stride(-1),
            output.stride(-2), output.stride(-1),
            batch_size, in_features, out_features,
            BLOCK_SIZE
        )
    else:
        tanh_linear_kernel[grid](
            input, weight, None, output,
            input.stride(-2), weight.stride(-2), weight.stride(-1),
            output.stride(-2), output.stride(-1),
            batch_size, in_features, out_features,
            BLOCK_SIZE
        )
    
    return output.view(*input_shape[:-1], out_features)

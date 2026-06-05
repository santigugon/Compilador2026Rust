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
    
    output_row = output_ptr + row * output_row_stride
    input_row = input_ptr + row * input_row_stride
    
    for i in range(0, out_features, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < out_features
        
        # Compute linear transformation
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, n_cols, BLOCK_SIZE):
            input_cols = j + tl.arange(0, BLOCK_SIZE)
            input_mask = input_cols < n_cols
            
            input_vals = tl.load(input_ptr + row * input_row_stride + input_cols, mask=input_mask, other=0.0)
            weight_vals = tl.load(weight_ptr + cols * weight_col_stride + j, mask=mask & input_mask, other=0.0)
            acc += tl.sum(input_vals * weight_vals, axis=0)
        
        # Add bias if present
        if bias_ptr is not None:
            bias_vals = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            acc += bias_vals
        
        # Apply tanh activation
        acc = tl.tanh(acc)
        
        # Store result
        tl.store(output_row + cols, acc, mask=mask)

def tanh_linear(input, weight, bias=None):
    assert input.dim() >= 2
    assert weight.dim() == 2
    assert input.size(-1) == weight.size(1)
    
    if bias is not None:
        assert bias.dim() == 1
        assert bias.size(0) == weight.size(0)
    
    input_shape = input.shape
    batch_size = 1
    for dim in input_shape[:-1]:
        batch_size *= dim
    
    in_features = input_shape[-1]
    out_features = weight.shape[0]
    
    input_reshaped = input.view(-1, in_features)
    
    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    grid = (batch_size, 1)
    
    # Launch kernel
    tanh_linear_kernel[grid](
        input_reshaped,
        weight,
        bias,
        output,
        input_reshaped.stride(0),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        batch_size,
        in_features,
        out_features,
        BLOCK_SIZE
    )
    
    return output.view(*input_shape[:-1], out_features)

import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, threshold, beta,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    
    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride
    
    for col in range(0, n_cols, BLOCK_SIZE):
        col_offset = col + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < n_cols
        
        input_vals = tl.load(input_row + col_offset * 1, mask=mask)
        weight_vals = tl.load(weight_ptr + col_offset * weight_col_stride, mask=mask)
        
        # Linear transformation
        linear_result = tl.sum(input_vals * weight_vals)
        
        # Softplus with threshold
        scaled_linear = linear_result * beta
        softplus_val = tl.log(1.0 + tl.exp(scaled_linear))
        softplus_val = tl.where(scaled_linear > threshold, linear_result, softplus_val)
        
        # Add bias if present
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + col_offset * 1, mask=mask)
            softplus_val += bias_val
        
        tl.store(output_row + col_offset * output_col_stride, softplus_val, mask=mask)

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    input = input.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    n_rows, n_cols = input.shape
    output = torch.empty(n_rows, weight.shape[0], dtype=input.dtype, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_rows, 1),)
    
    softplus_linear_kernel[grid](
        input, weight, bias, output,
        input.stride(0), weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        n_rows, n_cols, threshold, beta,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

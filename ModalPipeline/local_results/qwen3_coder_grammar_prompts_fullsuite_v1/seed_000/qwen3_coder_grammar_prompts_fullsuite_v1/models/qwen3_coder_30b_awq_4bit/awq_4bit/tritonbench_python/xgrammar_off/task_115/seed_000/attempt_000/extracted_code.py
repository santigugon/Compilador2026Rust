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
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    input_row = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride
    
    for col_idx in range(0, out_features, BLOCK_SIZE):
        col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < out_features
        
        output_offsets = col_idx * output_col_stride
        output_ptrs = output_row + output_offsets
        
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for k in range(0, n_cols, BLOCK_SIZE):
            k_offsets = k + tl.arange(0, BLOCK_SIZE)
            input_mask = k_offsets < n_cols
            
            input_ptrs = input_row + k_offsets
            input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)
            
            weight_row = weight_ptr + col_idx * weight_row_stride + k * weight_col_stride
            weight_ptrs = weight_row + k_offsets
            weight_vals = tl.load(weight_ptrs, mask=input_mask, other=0.0)
            
            acc += input_vals * weight_vals
        
        if bias_ptr is not None:
            bias_row = bias_ptr + col_idx
            bias_vals = tl.load(bias_row, mask=mask, other=0.0)
            acc += bias_vals
        
        output_vals = tl.tanh(acc)
        tl.store(output_ptrs, output_vals, mask=mask)

def tanh_linear(input, weight, bias=None) -> torch.Tensor:
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    input_shape = input.shape
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    
    output = torch.empty(input_shape[:-1] + (out_features,), dtype=input.dtype, device=input.device)
    
    n_rows = input_shape[:-1].numel()
    n_cols = in_features
    
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    
    tanh_linear_kernel[grid](
        input, weight, bias, output,
        input.stride(-1), weight.stride(0), weight.stride(1),
        output.stride(-1), output.stride(-1),
        n_rows, n_cols, out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

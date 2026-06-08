import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, other_ptr, output_ptr,
    input_row_stride, other_row_stride, output_row_stride,
    n_cols, n_rows,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    input_row_start = input_ptr + row_idx * input_row_stride
    other_row_start = other_ptr + row_idx * other_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = col_offsets < n_cols
    
    input_ptrs = input_row_start + col_offsets
    other_ptrs = other_row_start + col_offsets
    output_ptrs = output_row_start + col_offsets
    
    input_vals = tl.load(input_ptrs, mask=row_mask, other=0.0)
    
    # Compute softmax
    max_val = tl.max(input_vals, axis=0)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Multiply with other
    other_vals = tl.load(other_ptrs, mask=row_mask, other=0.0)
    result_vals = softmax_vals * other_vals
    
    # Store result
    tl.store(output_ptrs, result_vals, mask=row_mask)

def softmax_mul(input, other, dim, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and other are contiguous
    input = input.contiguous()
    other = other.contiguous()
    
    # Get dimensions
    n_rows, n_cols = input.shape[0], input.shape[1]
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1, 1)
    
    # Determine appropriate dtype for kernel
    if input.dtype == torch.float32:
        triton_dtype = tl.float32
    elif input.dtype == torch.float16:
        triton_dtype = tl.float16
    else:
        triton_dtype = tl.float32
    
    softmax_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        input_row_stride=input.stride(0),
        other_row_stride=other.stride(0),
        output_row_stride=out.stride(0),
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype
    )
    
    return out

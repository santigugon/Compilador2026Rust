import torch
import triton
import triton.language as tl

@triton.jit
def softmax_mul_kernel(
    input_ptr, other_ptr, output_ptr,
    input_row_stride, other_row_stride, output_row_stride,
    n_cols, dim, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input data
    input_ptrs = input_ptr + row * input_row_stride + col_offsets
    input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Apply softmax
    # Find max for numerical stability
    max_val = tl.max(input_vals, axis=0)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Load other data
    other_ptrs = other_ptr + row * other_row_stride + col_offsets
    other_vals = tl.load(other_ptrs, mask=mask, other=0.0)
    
    # Multiply softmax with other
    result = softmax_vals * other_vals
    
    # Store result
    output_ptrs = output_ptr + row * output_row_stride + col_offsets
    tl.store(output_ptrs, result, mask=mask)

def softmax_mul(input, other, dim, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and other have the same shape
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    if other.dim() == 0:
        other = other.expand(input.shape)
    elif other.shape != input.shape:
        # Handle broadcasting
        other = other.expand(input.shape)
    
    # Get dimensions
    n_rows, n_cols = input.shape[0], input.shape[1]
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1, 1)
    
    softmax_mul_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        input_row_stride=input.stride(0),
        other_row_stride=other.stride(0),
        output_row_stride=out.stride(0),
        n_cols=n_cols,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

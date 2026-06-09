import torch
import triton
import triton.language as tl

def _elu_linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_rows, n_cols, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return
    
    # Load row from input
    row_ptr = x_ptr + row * n_cols
    x = tl.load(row_ptr + tl.arange(0, BLOCK), mask=row_ptr + tl.arange(0, BLOCK) < x_ptr + n_rows * n_cols, other=0.0)
    
    # Compute linear transformation
    linear_result = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n_cols):
        w_val = tl.load(w_ptr + i * n_cols + row)
        linear_result += x[i] * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + row)
        linear_result += bias_val
    
    # Apply ELU
    elu_result = tl.where(linear_result > 0, linear_result, alpha * (tl.exp(linear_result) - 1.0))
    
    # Store result
    tl.store(out_ptr + row * n_cols + tl.arange(0, BLOCK), elu_result, mask=out_ptr + row * n_cols + tl.arange(0, BLOCK) < out_ptr + n_rows * n_cols)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Handle the case where input is a scalar
    if input.dim() == 0:
        input = input.unsqueeze(0)
    
    # Get dimensions
    n_rows = input.shape[0]
    n_cols = input.shape[1]
    
    # Ensure weight is transposed for correct computation
    weight_t = weight.t()
    
    # Create output tensor
    if inplace:
        out = input
    else:
        out = torch.empty(n_rows, weight.shape[0], dtype=input.dtype, device=input.device)
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n_rows, block),)
    
    # Handle bias
    b_ptr = bias if bias is not None else None
    
    _elu_linear_kernel[grid](input, weight_t, b_ptr, out, n_rows, n_cols, alpha, BLOCK=block)
    
    return out
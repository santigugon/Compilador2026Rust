import torch
import triton
import triton.language as tl

def softmax_mul(input, other, dim, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same device and dtype as input
    if other.device != input.device:
        other = other.to(input.device)
    if other.dtype != input.dtype:
        other = other.to(input.dtype)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        
    # Get dimensions
    input_shape = input.shape
    dim_size = input_shape[dim]
    
    # Flatten input and other to 2D for easier processing
    # We'll compute softmax along the specified dimension
    if dim == -1:
        new_shape = (1, -1)
    else:
        new_shape = (1, -1) if dim == 0 else (-1, dim_size)
    
    # Reshape for processing
    input_reshaped = input.view(-1, dim_size)
    other_reshaped = other.view(-1, dim_size)
    out_reshaped = out.view(-1, dim_size)
    
    # Process each row
    n_rows = input_reshaped.shape[0]
    block = 256
    grid = (n_rows,)
    
    # Apply softmax and multiply
    _softmax_mul_kernel[grid](input_reshaped, other_reshaped, out_reshaped, dim_size, BLOCK=block)
    
    return out

@triton.jit
def _softmax_mul_kernel(input_ptr, other_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Load input row
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input and other values
    input_row = tl.load(input_ptr + pid * n + offsets, mask=mask, other=0.0)
    other_row = tl.load(other_ptr + pid * n + offsets, mask=mask, other=0.0)
    
    # Compute softmax
    # Subtract max for numerical stability
    max_val = tl.max(input_row, axis=0)
    input_shifted = input_row - max_val
    
    # Compute exp
    exp_vals = tl.exp(input_shifted)
    
    # Compute sum
    sum_vals = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / sum_vals
    
    # Multiply by other
    result = softmax_vals * other_row
    
    # Store result
    tl.store(out_ptr + pid * n + offsets, result, mask=mask)
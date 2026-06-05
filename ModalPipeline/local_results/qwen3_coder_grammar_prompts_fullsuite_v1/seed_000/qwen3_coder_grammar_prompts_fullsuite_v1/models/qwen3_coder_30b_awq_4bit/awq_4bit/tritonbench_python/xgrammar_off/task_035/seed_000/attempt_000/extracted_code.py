import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr,
    vec_stride_0: tl.constexpr,
    other_stride_0: tl.constexpr,
    out_stride_0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program processes one row of the input matrix
    row = pid
    if row >= n:
        return
    
    # Initialize output row
    out_row = tl.full((m,), 0.0, dtype=tl.float32)
    
    # Compute matrix-vector multiplication: row * vec
    for i in range(0, m, BLOCK_SIZE):
        vec_offsets = i + tl.arange(0, BLOCK_SIZE)
        vec_mask = vec_offsets < m
        vec_vals = tl.load(vec_ptr + vec_offsets * vec_stride_0, mask=vec_mask, other=0.0)
        
        input_offsets = row * input_stride_0 + vec_offsets * input_stride_1
        input_mask = vec_offsets < m
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        out_row += input_vals * vec_vals
    
    # Apply sigmoid to the result
    out_row = 1.0 / (1.0 + tl.exp(-out_row))
    
    # Apply subtraction with scaling
    other_val = tl.load(other_ptr)
    out_row = out_row - other_val * 1.0  # alpha is 1.0 by default
    
    # Store the result
    out_offsets = row * out_stride_0
    tl.store(out_ptr + out_offsets, out_row, mask=tl.arange(0, m) < m)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Validate input shapes
    n, m = input.shape
    assert vec.shape == (m,), f"Expected vec shape ({m},), got {vec.shape}"
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(n, dtype=torch.float32, device=input.device)
    else:
        assert out.shape == (n,), f"Expected out shape ({n},), got {out.shape}"
    
    # Launch kernel
    block_size = 256
    grid_size = triton.cdiv(n, 1)  # One program per row
    
    # Create a temporary tensor for the intermediate result
    temp = torch.empty(n, dtype=torch.float32, device=input.device)
    
    # Compute matrix-vector multiplication + sigmoid + subtraction
    _fused_mv_sigmoid_sub_kernel[grid_size](
        input, vec, other, temp,
        n, m,
        input.stride(0), input.stride(1),
        vec.stride(0),
        other.stride(0) if torch.is_tensor(other) else 0,
        temp.stride(0),
        BLOCK_SIZE=block_size
    )
    
    # Apply final subtraction with alpha scaling
    if torch.is_tensor(other):
        out = temp - alpha * other
    else:
        out = temp - alpha * other
    
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_norm_kernel(y_ptr, x_ptr, out_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    y_new = y + alpha * x
    tl.store(y_ptr + offsets, y_new, mask=mask)
    
    # Compute squared sum for 2-norm
    y_sq = y_new * y_new
    tl.store(out_ptr + offsets, y_sq, mask=mask)

@triton.jit
def reduce_sum_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    vals = tl.load(in_ptr + offsets, mask=mask)
    sum_val = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, sum_val)

@triton.jit
def sqrt_kernel(out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    val = tl.load(out_ptr + offsets, mask=mask)
    sqrt_val = tl.sqrt(val)
    tl.store(out_ptr + offsets, sqrt_val, mask=mask)

def scaled_add_norm(y, x, alpha):
    assert y.shape == x.shape, "y and x must have the same shape"
    assert len(y.shape) == 1, "y and x must be 1-dimensional"
    
    n = y.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor for squared values
    squared_y = torch.zeros_like(y)
    
    # First kernel: compute y += alpha * x and store squared values
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scaled_add_norm_kernel[grid](y, x, squared_y, n, alpha, BLOCK_SIZE)
    
    # Second kernel: reduce sum of squared values
    reduced_sum = torch.zeros(triton.cdiv(n, BLOCK_SIZE), dtype=torch.float32, device=y.device)
    reduce_sum_kernel[grid](squared_y, reduced_sum, n, BLOCK_SIZE)
    
    # Third kernel: compute square root of the sum
    final_result = torch.sum(reduced_sum)
    result_tensor = torch.tensor([final_result], dtype=torch.float32, device=y.device)
    sqrt_kernel[1](result_tensor, 1, BLOCK_SIZE)
    
    return result_tensor[0]
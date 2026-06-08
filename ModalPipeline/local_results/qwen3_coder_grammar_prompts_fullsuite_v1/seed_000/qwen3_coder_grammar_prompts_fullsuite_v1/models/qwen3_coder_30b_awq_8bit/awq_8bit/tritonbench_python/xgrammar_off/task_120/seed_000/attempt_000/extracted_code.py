import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Each program processes one row of A
    row = pid
    if row >= n:
        return
    
    # Compute dot product of row with x
    acc = 0.0
    for i in range(0, m, BLOCK_SIZE):
        # Load x elements with masking
        x_offsets = i + tl.arange(0, BLOCK_SIZE)
        x_mask = x_offsets < m
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        
        # Load A[row, :] elements with masking
        A_offsets = row * m + x_offsets
        A_mask = x_offsets < m
        A_vals = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(A_vals * x_vals)
    
    # Compute y[row] = alpha * mv(A, x)[row] + beta * y[row]
    y_val = tl.load(y_ptr + row)
    new_y_val = alpha * acc + beta * y_val
    tl.store(y_ptr + row, new_y_val)
    
    # Store intermediate result for final dot product
    tl.store(out_ptr + row, new_y_val)

@triton.jit
def _dot_product_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute dot product of y and x
    acc = 0.0
    for i in range(0, n, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(y_vals * x_vals)
    
    # Store final result
    tl.store(out_ptr, acc)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Validate input shapes
    assert A.dim() == 2, "A must be a 2D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    assert A.shape[1] == x.shape[0], "A's second dimension must match x's dimension"
    assert A.shape[0] == y.shape[0], "A's first dimension must match y's dimension"
    
    n, m = A.shape
    # Create temporary tensor for intermediate y values
    y_temp = torch.empty_like(y)
    y_temp.copy_(y)
    
    # Create output tensor for final dot product
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Launch kernel to compute y = alpha * mv(A, x) + beta * y
    block_size = 256
    grid_size = triton.cdiv(n, block_size)
    _matrix_vector_dot_kernel[grid_size](A, x, y_temp, y_temp, n, m, alpha, beta, BLOCK_SIZE=block_size)
    
    # Launch kernel to compute dot product of updated y and x
    _dot_product_kernel[1](y_temp, x, out, n, BLOCK_SIZE=block_size)
    
    return out

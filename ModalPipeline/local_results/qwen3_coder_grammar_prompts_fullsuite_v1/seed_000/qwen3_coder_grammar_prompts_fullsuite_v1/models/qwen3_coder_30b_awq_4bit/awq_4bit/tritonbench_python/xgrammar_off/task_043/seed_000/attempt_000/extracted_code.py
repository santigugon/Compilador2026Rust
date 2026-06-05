import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mv_norm_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load x
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute matrix-vector product A * x
    y_vals = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n):
        a_row = tl.load(A_ptr + i * n + offsets, mask=mask, other=0.0)
        x_val = tl.load(x_ptr + i, other=0.0)
        y_vals += a_row * x_val
    
    # Apply scaling and addition: y = alpha * (A * x) + beta * y
    y_vals = alpha * y_vals + beta * tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, y_vals, mask=mask)
    
    # Compute norm
    if p == 1.0:
        abs_vals = tl.abs(y_vals)
        sum_vals = tl.sum(abs_vals, axis=0)
    elif p == 2.0:
        square_vals = y_vals * y_vals
        sum_vals = tl.sum(square_vals, axis=0)
        sum_vals = tl.sqrt(sum_vals)
    elif p == float('inf'):
        abs_vals = tl.abs(y_vals)
        sum_vals = tl.max(abs_vals, axis=0)
    else:
        abs_vals = tl.abs(y_vals)
        pow_vals = tl.pow(abs_vals, p)
        sum_vals = tl.sum(pow_vals, axis=0)
        sum_vals = tl.pow(sum_vals, 1.0 / p)
    
    # Store the norm
    tl.store(out_ptr, sum_vals, mask=mask)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    x = x.contiguous()
    
    # Validate dimensions
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert x.dim() == 1 and x.shape[0] == A.shape[0], "x must be a vector with length equal to A's dimension"
    
    n = A.shape[0]
    y = torch.empty_like(x)
    
    # Initialize y with zeros
    y.zero_()
    
    # Compute matrix-vector product and apply scaling
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create a temporary tensor for the result of A * x
    temp_y = torch.empty(n, dtype=torch.float32, device=A.device)
    
    # First compute A * x
    _symmetric_mv_norm_kernel[grid](A, x, y, temp_y, n, alpha, beta, p, BLOCK=block)
    
    # Then compute the norm
    if p == 1.0:
        norm = torch.sum(torch.abs(temp_y))
    elif p == 2.0:
        norm = torch.sqrt(torch.sum(temp_y * temp_y))
    elif p == float('inf'):
        norm = torch.max(torch.abs(temp_y))
    else:
        norm = torch.pow(torch.sum(torch.pow(torch.abs(temp_y), p)), 1.0 / p)
    
    return norm

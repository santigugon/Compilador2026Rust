import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mv_kernel(A_ptr, x_ptr, y_ptr, n: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Initialize y with beta * y
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y = beta * y
    
    # Compute matrix-vector product with symmetric matrix
    for i in range(n):
        # Load x[i]
        x_i = tl.load(x_ptr + i, other=0.0)
        
        # Load A[i, :] and compute dot product with x
        a_row = tl.load(A_ptr + i * n + offsets, mask=mask, other=0.0)
        
        # Accumulate the result
        y = y + alpha * x_i * a_row
    
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def _norm_kernel(y_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load y values
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute |y|^p
    y_p = tl.abs(y) ** p
    
    # Sum up all |y|^p values
    y_p_sum = tl.sum(y_p, axis=0)
    
    # Store the result
    tl.store(out_ptr + 0, y_p_sum ** (1.0 / p), mask=True)


def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    x = x.contiguous()
    
    # Get dimensions
    n = A.shape[0]
    
    # Initialize output tensor
    y = torch.empty_like(x)
    
    # Initialize y with x
    y = x.clone()
    
    # Compute y = alpha * mv(A, x) + beta * y
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # First compute the matrix-vector product
    _symmetric_mv_kernel[grid](A, x, y, n, alpha, beta, BLOCK=block)
    
    # Compute the norm
    out = torch.empty(1, dtype=torch.float32)
    _norm_kernel[grid](y, out, n, p, BLOCK=block)
    
    return out
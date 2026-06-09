import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mv_norm_kernel(A_ptr, x_ptr, y_ptr, output_ptr, n, alpha, beta, p, stride_A, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Load x vector
    x_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x_mask = x_offsets < n
    x_vals = tl.load(x_ptr + x_offsets, mask=x_mask)
    
    # Compute matrix-vector product for this block
    y_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    y_mask = y_offsets < n
    
    # Initialize y with beta * y
    y_vals = tl.load(y_ptr + y_offsets, mask=y_mask)
    y_vals = alpha * y_vals + beta * y_vals
    
    # Compute A * x for this block
    for i in range(n):
        a_val = tl.load(A_ptr + i * stride_A + block_start + tl.arange(0, BLOCK_SIZE), mask=(block_start + tl.arange(0, BLOCK_SIZE)) < n)
        x_val = tl.load(x_ptr + i)
        y_vals += a_val * x_val
    
    # Store result back to y
    tl.store(y_ptr + y_offsets, y_vals, mask=y_mask)
    
    # Compute norm
    if pid == 0:
        # Reduction to compute norm
        y_vals = tl.load(y_ptr + tl.arange(0, n))
        if p == 2.0:
            # Euclidean norm
            norm = tl.sqrt(tl.sum(y_vals * y_vals))
        elif p == 1.0:
            # L1 norm
            norm = tl.sum(tl.abs(y_vals))
        else:
            # General Lp norm
            norm = tl.pow(tl.sum(tl.pow(tl.abs(y_vals), p)), 1.0 / p)
        tl.store(output_ptr, norm)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert x.dim() == 1 and x.shape[0] == A.shape[0], "x must be a vector with length equal to A's dimension"
    
    n = A.shape[0]
    y = torch.zeros_like(x)
    output = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    _symmetric_mv_norm_kernel[grid](
        A, x, y, output,
        n, alpha, beta, p,
        A.stride(0), x.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A, L, n, BLOCK_SIZE: tl.constexpr):
    # Compute Cholesky decomposition
    for k in range(n):
        # Compute diagonal element
        sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for p in range(k):
            sum_val += L[k, p] * L[k, p]
        L[k, k] = tl.sqrt(A[k, k] - sum_val)
        
        # Compute off-diagonal elements
        for i in range(k + 1, n):
            sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for p in range(k):
                sum_val += L[i, p] * L[k, p]
            L[i, k] = (A[i, k] - sum_val) / L[k, k]

@triton.jit
def solve_lower_triangular_kernel(L, b, x, n, k, BLOCK_SIZE: tl.constexpr):
    # Solve L * x = b
    for i in range(n):
        sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(i):
            sum_val += L[i, j] * x[j]
        x[i] = (b[i] - sum_val) / L[i, i]

@triton.jit
def solve_upper_triangular_kernel(L, b, x, n, k, BLOCK_SIZE: tl.constexpr):
    # Solve L.T * x = b (upper triangular solve)
    for i in range(n - 1, -1, -1):
        sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(i + 1, n):
            sum_val += L[j, i] * x[j]
        x[i] = (b[i] - sum_val) / L[i, i]

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, k = b.shape
    assert A.shape == (n, n), "Matrix A must be n x n"
    
    # Allocate output tensor
    x = torch.zeros_like(b)
    
    # Create output tensor for L
    L = torch.zeros_like(A)
    
    # Compute Cholesky decomposition
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE), 1)
    cholesky_kernel[grid](A, L, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Solve L * x = b
    grid = (triton.cdiv(n, BLOCK_SIZE), 1)
    solve_lower_triangular_kernel[grid](L, b, x, n, k, BLOCK_SIZE=BLOCK_SIZE)
    
    # Solve L.T * x = result (upper triangular solve)
    grid = (triton.cdiv(n, BLOCK_SIZE), 1)
    solve_upper_triangular_kernel[grid](L, x, x, n, k, BLOCK_SIZE=BLOCK_SIZE)
    
    return x

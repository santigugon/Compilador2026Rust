import torch
import triton
import triton.language as tl

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Check if A is square
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    # Check if b has compatible dimensions
    assert b.dim() == 2 and b.shape[0] == A.shape[0], "b must have compatible dimensions with A"
    
    n, k = A.shape[0], b.shape[1]
    
    # Perform Cholesky decomposition
    L = torch.linalg.cholesky(A)
    
    # Solve L * y = b for y
    y = torch.triangular_solve(b, L, upper=False)[0]
    
    # Solve L.T * x = y for x
    x = torch.triangular_solve(y, L.T, upper=True)[0]
    
    return x
import torch
import triton
import triton.language as tl

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix A must be square"
    assert b.shape == (n,), "Vector b must have shape (n,)"
    
    # Perform LU decomposition
    L, U, P = torch.lu_unpack(torch.lu(A), unpack_data=False)
    
    # Solve L @ y = P @ b
    Pb = torch.mm(P, b.unsqueeze(1)).squeeze(1)
    y = torch.zeros_like(b)
    for i in range(n):
        y[i] = Pb[i] - sum(L[i][j] * y[j] for j in range(i))
    
    # Solve U @ x = y
    x = torch.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    
    return x
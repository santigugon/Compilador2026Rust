import torch
import triton
import triton.language as tl

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, k = A.shape[0], b.shape[1]
    
    # Perform Cholesky decomposition
    L = torch.linalg.cholesky(A)
    
    # Solve L * y = b for y
    y = torch.trtrs(b, L, upper=False)[0]
    
    # Solve L.T * x = y for x
    x = torch.trtrs(y, L.T, upper=True)[0]
    
    return x
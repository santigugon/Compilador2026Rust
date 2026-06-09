import torch
import triton
import triton.language as tl

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1]
    
    # Perform QR decomposition using torch
    Q, R = torch.linalg.qr(A)
    
    # Solve Rx = Q^T b
    # First compute Q^T b
    Qb = torch.matmul(Q.transpose(0, 1), b)
    
    # Then solve Rx = Q^T b using triangular solver
    x = torch.triangular_solve(Qb, R, upper=True)[0]
    
    return x
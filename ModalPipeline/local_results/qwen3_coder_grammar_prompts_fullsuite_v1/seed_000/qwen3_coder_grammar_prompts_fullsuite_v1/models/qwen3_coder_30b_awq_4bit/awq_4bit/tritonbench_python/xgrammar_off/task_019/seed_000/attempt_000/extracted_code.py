import torch
import triton
import triton.language as tl

@triton.jit
def lu_decompose_kernel(A, L, U, P, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return
    
    # Initialize L, U, P
    for i in range(n):
        L[i, pid] = 0.0
        U[pid, i] = A[pid, i]
        P[pid] = pid
    
    # LU decomposition
    for k in range(n):
        # Find pivot
        max_val = 0.0
        max_idx = k
        for i in range(k, n):
            val = tl.abs(U[i, k])
            if val > max_val:
                max_val = val
                max_idx = i
        
        # Swap rows in U and P
        if max_idx != k:
            for j in range(n):
                temp = U[k, j]
                U[k, j] = U[max_idx, j]
                U[max_idx, j] = temp
            
            temp = P[k]
            P[k] = P[max_idx]
            P[max_idx] = temp
        
        # Compute L and U
        for i in range(k + 1, n):
            if U[k, k] != 0.0:
                L[i, k] = U[i, k] / U[k, k]
                for j in range(k + 1, n):
                    U[i, j] = U[i, j] - L[i, k] * U[k, j]
            else:
                L[i, k] = 0.0

@triton.jit
def forward_substitution_kernel(L, b, x, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return
    
    # Forward substitution: L @ y = b
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * x[j]
        x[i] = (b[i] - sum_val) / L[i, i]

@triton.jit
def backward_substitution_kernel(U, y, x, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return
    
    # Backward substitution: U @ x = y
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U[i, j] * x[j]
        x[i] = (y[i] - sum_val) / U[i, i]

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    device = A.device
    
    # Allocate memory for L, U, P
    L = torch.zeros((n, n), device=device, dtype=torch.float32)
    U = torch.zeros((n, n), device=device, dtype=torch.float32)
    P = torch.zeros(n, device=device, dtype=torch.int32)
    
    # Copy A to U
    U.copy_(A)
    
    # Launch LU decomposition kernel
    BLOCK_SIZE = 16
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    lu_decompose_kernel[grid](U, L, U, P, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Forward substitution
    y = torch.zeros(n, device=device, dtype=torch.float32)
    y.copy_(b)
    forward_substitution_kernel[grid](L, b, y, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Backward substitution
    x = torch.zeros(n, device=device, dtype=torch.float32)
    backward_substitution_kernel[grid](U, y, x, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return x

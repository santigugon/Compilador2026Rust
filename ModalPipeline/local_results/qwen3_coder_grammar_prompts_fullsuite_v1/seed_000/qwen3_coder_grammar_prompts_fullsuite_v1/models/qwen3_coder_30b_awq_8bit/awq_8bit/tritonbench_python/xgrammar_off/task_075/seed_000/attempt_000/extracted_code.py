import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A, L, n, BLOCK_SIZE: tl.constexpr):
    # Compute Cholesky decomposition
    for k in range(n):
        # Compute diagonal element
        if k < n:
            sum_val = tl.zeros([1], dtype=tl.float32)
            for p in range(k):
                sum_val += L[k, p] * L[k, p]
            L[k, k] = tl.sqrt(A[k, k] - sum_val)
        
        # Compute off-diagonal elements
        for i in range(k + 1, n):
            if i < n and k < n:
                sum_val = tl.zeros([1], dtype=tl.float32)
                for p in range(k):
                    sum_val += L[i, p] * L[k, p]
                L[i, k] = (A[i, k] - sum_val) / L[k, k]

@triton.jit
def forward_substitution_kernel(L, b, x, n, k, BLOCK_SIZE: tl.constexpr):
    # Forward substitution: L * y = b
    for i in range(n):
        if i < n:
            sum_val = tl.zeros([1], dtype=tl.float32)
            for j in range(i):
                sum_val += L[i, j] * x[j]
            x[i] = (b[i] - sum_val) / L[i, i]

@triton.jit
def backward_substitution_kernel(L, x, y, n, k, BLOCK_SIZE: tl.constexpr):
    # Backward substitution: L.T * x = y
    for i in range(n - 1, -1, -1):
        if i < n:
            sum_val = tl.zeros([1], dtype=tl.float32)
            for j in range(i + 1, n):
                sum_val += L[j, i] * x[j]
            x[i] = (y[i] - sum_val) / L[i, i]

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, k = A.shape[0], b.shape[1]
    A = A.float()
    b = b.float()
    
    # Allocate output tensor
    x = torch.empty(n, k, dtype=torch.float32, device=A.device)
    
    # Create workspace for L
    L = torch.empty(n, n, dtype=torch.float32, device=A.device)
    
    # Copy A to L
    L.copy_(A)
    
    # Compute Cholesky decomposition
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    cholesky_kernel[grid](L, L, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Solve for x using forward and backward substitution
    for i in range(k):
        # Forward substitution
        y = torch.empty(n, dtype=torch.float32, device=A.device)
        b_col = b[:, i]
        forward_substitution_kernel[grid](L, b_col, y, n, 1, BLOCK_SIZE=BLOCK_SIZE)
        
        # Backward substitution
        backward_substitution_kernel[grid](L, y, x[:, i], n, 1, BLOCK_SIZE=BLOCK_SIZE)
    
    return x

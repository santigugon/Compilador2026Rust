import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_kernel(A_ptr, L_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Compute Cholesky decomposition: A = L * L.T
    # L is lower triangular
    for j in range(n):
        # Compute diagonal element
        sum_val = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(j):
            sum_val += L_ptr[j * n + k] * L_ptr[j * n + k]
        diag_val = A_ptr[j * n + j] - sum_val
        L_ptr[j * n + j] = tl.sqrt(diag_val)
        
        # Compute off-diagonal elements
        for i in range(j + 1, n):
            sum_val = tl.zeros([BLOCK], dtype=tl.float32)
            for k in range(j):
                sum_val += L_ptr[i * n + k] * L_ptr[j * n + k]
            L_ptr[i * n + j] = (A_ptr[i * n + j] - sum_val) / L_ptr[j * n + j]

@triton.jit
def _forward_substitution_kernel(L_ptr, b_ptr, x_ptr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Solve L * x = b using forward substitution
    for i in range(n):
        sum_val = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(i):
            sum_val += L_ptr[i * n + j] * x_ptr[j * k + 0]  # Assuming k=1 for simplicity
        x_ptr[i * k + 0] = (b_ptr[i * k + 0] - sum_val) / L_ptr[i * n + i]

@triton.jit
def _backward_substitution_kernel(L_ptr, x_ptr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Solve L.T * x = b using backward substitution
    for i in range(n - 1, -1, -1):
        sum_val = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(i + 1, n):
            sum_val += L_ptr[j * n + i] * x_ptr[j * k + 0]  # Assuming k=1 for simplicity
        x_ptr[i * k + 0] = (x_ptr[i * k + 0] - sum_val) / L_ptr[i * n + i]

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    A = A.contiguous()
    b = b.contiguous()
    
    # Check dimensions
    n, n = A.shape
    _, k = b.shape
    
    # Allocate output tensor
    x = torch.empty_like(b)
    
    # Create a copy of A for Cholesky decomposition
    L = torch.empty_like(A)
    
    # Compute Cholesky decomposition
    block = 256
    grid = (triton.cdiv(n, block),)
    _cholesky_kernel[grid](A, L, n, BLOCK=block)
    
    # Copy b to x for solving
    x.copy_(b)
    
    # Forward substitution: L * y = b
    # For each column of b
    for col in range(k):
        # Forward substitution
        _forward_substitution_kernel[grid](L, b, x, n, k, BLOCK=block)
    
    # Backward substitution: L.T * x = y
    # For each column of x
    for col in range(k):
        # Backward substitution
        _backward_substitution_kernel[grid](L, x, n, k, BLOCK=block)
    
    return x

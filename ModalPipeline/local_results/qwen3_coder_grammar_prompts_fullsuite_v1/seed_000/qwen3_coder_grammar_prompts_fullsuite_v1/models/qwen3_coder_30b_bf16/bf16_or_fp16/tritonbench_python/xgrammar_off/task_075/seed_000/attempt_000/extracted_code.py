import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A_ptr, L_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Compute Cholesky decomposition L * L.T = A
    for k in range(n):
        # Compute diagonal element
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for p in range(k):
            acc += tl.load(L_ptr + p * n + k) * tl.load(L_ptr + p * n + k)
        diag = tl.load(A_ptr + k * n + k) - acc
        diag = tl.sqrt(diag)
        tl.store(L_ptr + k * n + k, diag)
        
        # Compute off-diagonal elements
        for i in range(k + 1, n):
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for p in range(k):
                acc += tl.load(L_ptr + p * n + k) * tl.load(L_ptr + p * n + i)
            val = (tl.load(A_ptr + k * n + i) - acc) / diag
            tl.store(L_ptr + k * n + i, val)

@triton.jit
def forward_substitution_kernel(L_ptr, b_ptr, x_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    # Solve L * y = b for y
    for i in range(n):
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(i):
            acc += tl.load(L_ptr + j * n + i) * tl.load(b_ptr + j * k + 0)
        val = (tl.load(b_ptr + i * k + 0) - acc) / tl.load(L_ptr + i * n + i)
        tl.store(b_ptr + i * k + 0, val)

@triton.jit
def backward_substitution_kernel(L_ptr, b_ptr, x_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    # Solve L.T * x = y for x
    for i in range(n - 1, -1, -1):
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(i + 1, n):
            acc += tl.load(L_ptr + i * n + j) * tl.load(b_ptr + j * k + 0)
        val = (tl.load(b_ptr + i * k + 0) - acc) / tl.load(L_ptr + i * n + i)
        tl.store(b_ptr + i * k + 0, val)

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, k = A.shape[0], b.shape[1]
    assert A.shape == (n, n), "A must be a square matrix"
    assert b.shape == (n, k), "b must be of shape (n, k)"
    
    # Allocate output tensor
    x = torch.empty_like(b)
    
    # Create a copy of A for Cholesky decomposition
    L = torch.empty_like(A)
    
    # Launch Cholesky kernel
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    cholesky_kernel[grid](A, L, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Copy b to x for solving
    x.copy_(b)
    
    # Forward substitution: L * y = b
    forward_substitution_kernel[grid](L, x, x, n, k, BLOCK_SIZE=BLOCK_SIZE)
    
    # Backward substitution: L.T * x = y
    backward_substitution_kernel[grid](L, x, x, n, k, BLOCK_SIZE=BLOCK_SIZE)
    
    return x

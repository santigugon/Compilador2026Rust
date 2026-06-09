import torch
import triton
import triton.language as tl

@triton.jit
def _solve_multiple_lu_kernel(
    A_ptr, Bs_ptr, X_ptr,
    n, k, batch_size,
    pivot: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Load A matrix for this batch
    A_batch = A_ptr + batch_idx * n * n
    Bs_batch = Bs_ptr + batch_idx * n * k
    X_batch = X_ptr + batch_idx * n * k
    
    # Allocate shared memory for L and U matrices
    L = tl.shared_ptr(tl.float32, BLOCK_SIZE * BLOCK_SIZE)
    U = tl.shared_ptr(tl.float32, BLOCK_SIZE * BLOCK_SIZE)
    
    # Initialize L and U
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i * BLOCK_SIZE + j] = 1.0
            else:
                L[i * BLOCK_SIZE + j] = 0.0
            if i <= j:
                U[i * BLOCK_SIZE + j] = 0.0
            else:
                U[i * BLOCK_SIZE + j] = 0.0
    
    # LU decomposition with partial pivoting
    for i in range(n):
        # Find pivot
        if pivot:
            max_val = tl.abs(A_batch[i * n + i])
            pivot_row = i
            for row in range(i + 1, n):
                if tl.abs(A_batch[row * n + i]) > max_val:
                    max_val = tl.abs(A_batch[row * n + i])
                    pivot_row = row
            if pivot_row != i:
                # Swap rows in A
                for col in range(n):
                    temp = A_batch[i * n + col]
                    A_batch[i * n + col] = A_batch[pivot_row * n + col]
                    A_batch[pivot_row * n + col] = temp
    
        # Compute L and U
        for j in range(i + 1, n):
            factor = A_batch[j * n + i] / A_batch[i * n + i]
            L[j * BLOCK_SIZE + i] = factor
            for k_idx in range(i, n):
                A_batch[j * n + k_idx] -= factor * A_batch[i * n + k_idx]
    
    # Copy A to U
    for i in range(n):
        for j in range(i, n):
            U[i * BLOCK_SIZE + j] = A_batch[i * n + j]
    
    # Forward substitution to solve L * Y = B
    for i in range(n):
        for j in range(k):
            y = 0.0
            for l in range(i):
                y += L[i * BLOCK_SIZE + l] * Bs_batch[l * k + j]
            Bs_batch[i * k + j] = Bs_batch[i * k + j] - y
    
    # Backward substitution to solve U * X = Y
    for i in range(n - 1, -1, -1):
        for j in range(k):
            x = Bs_batch[i * k + j]
            for l in range(i + 1, n):
                x -= U[i * BLOCK_SIZE + l] * Bs_batch[l * k + j]
            Bs_batch[i * k + j] = x / U[i * BLOCK_SIZE + i]

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    # Validate inputs
    assert A.ndim >= 2, "A must have at least 2 dimensions"
    assert Bs.ndim >= 2, "Bs must have at least 2 dimensions"
    assert A.shape[-1] == A.shape[-2], "A must be square"
    assert A.shape[-1] == Bs.shape[-2], "A and Bs must have compatible dimensions"
    
    # Get batch dimensions
    batch_shape = A.shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    n = A.shape[-1]
    k = Bs.shape[-1]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(Bs)
    else:
        assert out.shape == Bs.shape, "out tensor must have the same shape as Bs"
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (batch_size, 1, 1)
    
    # Ensure tensors are on the same device and have correct dtype
    A = A.to(torch.float32)
    Bs = Bs.to(torch.float32)
    out = out.to(torch.float32)
    
    # Launch kernel
    _solve_multiple_lu_kernel[grid](
        A, Bs, out,
        n, k, batch_size,
        pivot=pivot,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

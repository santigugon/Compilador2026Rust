import torch
import triton
import triton.language as tl

@triton.jit
def _solve_multiple_lu_kernel(
    A_ptr, Bs_ptr, X_ptr, 
    stride_A_batch, stride_A_row, stride_A_col,
    stride_B_batch, stride_B_row, stride_B_col,
    stride_X_batch, stride_X_row, stride_X_col,
    batch_size, n, k,
    pivot: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Load A matrix for this batch
    A_batch = A_ptr + batch_idx * stride_A_batch
    Bs_batch = Bs_ptr + batch_idx * stride_B_batch
    X_batch = X_ptr + batch_idx * stride_X_batch
    
    # Allocate shared memory for L and U matrices
    L = tl.shared_ptr(tl.zeros((n, n), dtype=tl.float32), (n, n))
    U = tl.shared_ptr(tl.zeros((n, n), dtype=tl.float32), (n, n))
    
    # Copy A to L and U
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = tl.load(A_batch + i * stride_A_row + j * stride_A_col)
            else:
                L[i, j] = tl.load(A_batch + i * stride_A_row + j * stride_A_col)
    
    # LU decomposition with partial pivoting
    if pivot:
        for i in range(n):
            # Find pivot
            max_val = tl.abs(U[i, i])
            pivot_row = i
            for j in range(i + 1, n):
                if tl.abs(U[j, i]) > max_val:
                    max_val = tl.abs(U[j, i])
                    pivot_row = j
            
            # Swap rows in L and U
            if pivot_row != i:
                for j in range(n):
                    temp = U[i, j]
                    U[i, j] = U[pivot_row, j]
                    U[pivot_row, j] = temp
                    
                    if j < i:
                        temp = L[i, j]
                        L[i, j] = L[pivot_row, j]
                        L[pivot_row, j] = temp
            
            # Eliminate
            for j in range(i + 1, n):
                if U[i, i] != 0:
                    L[j, i] = U[j, i] / U[i, i]
                    for k_idx in range(i + 1, n):
                        U[j, k_idx] = U[j, k_idx] - L[j, i] * U[i, k_idx]
    
    # Forward substitution
    for i in range(n):
        for j in range(k):
            X_batch[i * stride_X_row + j * stride_X_col] = tl.load(Bs_batch + i * stride_B_row + j * stride_B_col)
    
    for i in range(n):
        for j in range(k):
            for l in range(i):
                X_batch[i * stride_X_row + j * stride_X_col] -= L[i, l] * X_batch[l * stride_X_row + j * stride_X_col]
    
    # Backward substitution
    for i in range(n - 1, -1, -1):
        for j in range(k):
            for l in range(i + 1, n):
                X_batch[i * stride_X_row + j * stride_X_col] -= U[i, l] * X_batch[l * stride_X_row + j * stride_X_col]
            if U[i, i] != 0:
                X_batch[i * stride_X_row + j * stride_X_col] /= U[i, i]

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    if A.dim() < 2:
        raise ValueError("A must have at least 2 dimensions")
    if Bs.dim() < 2:
        raise ValueError("Bs must have at least 2 dimensions")
    
    batch_dims_A = A.shape[:-2]
    batch_dims_Bs = Bs.shape[:-2]
    
    if batch_dims_A != batch_dims_Bs:
        raise ValueError("Batch dimensions of A and Bs must match")
    
    n_A, n_Bs = A.shape[-2], Bs.shape[-2]
    if n_A != n_Bs:
        raise ValueError("Last two dimensions of A and Bs must match")
    
    batch_size = 1
    for dim in batch_dims_A:
        batch_size *= dim
    
    n = n_A
    k = Bs.shape[-1]
    
    if out is None:
        out = torch.empty_like(Bs)
    
    if batch_size == 0:
        return out
    
    # Launch kernel
    grid = (batch_size,)
    BLOCK_SIZE = 32
    
    _solve_multiple_lu_kernel[grid](
        A, Bs, out,
        A.stride(-3) if A.dim() > 2 else 0,
        A.stride(-2),
        A.stride(-1),
        Bs.stride(-3) if Bs.dim() > 2 else 0,
        Bs.stride(-2),
        Bs.stride(-1),
        out.stride(-3) if out.dim() > 2 else 0,
        out.stride(-2),
        out.stride(-1),
        batch_size, n, k,
        pivot,
        BLOCK_SIZE
    )
    
    return out

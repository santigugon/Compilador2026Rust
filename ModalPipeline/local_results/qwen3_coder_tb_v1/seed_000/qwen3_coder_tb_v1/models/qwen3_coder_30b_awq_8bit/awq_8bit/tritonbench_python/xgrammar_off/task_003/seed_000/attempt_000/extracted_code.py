import torch
import triton
import triton.language as tl

@triton.jit
def _solve_multiple_lu_kernel(
    A_ptr, Bs_ptr, X_ptr, 
    stride_A_batch, stride_A_n, stride_A_k,
    stride_B_batch, stride_B_n, stride_B_k,
    stride_X_batch, stride_X_n, stride_X_k,
    batch_size, n, k,
    pivot: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    A_batch = A_ptr + batch_idx * stride_A_batch
    B_batch = Bs_ptr + batch_idx * stride_B_batch
    X_batch = X_ptr + batch_idx * stride_X_batch
    
    # Allocate shared memory for the LU decomposition
    L = tl.shared_ptr(tl.zeros((n, n), dtype=tl.float32), (n, n))
    U = tl.shared_ptr(tl.zeros((n, n), dtype=tl.float32), (n, n))
    P = tl.shared_ptr(tl.zeros((n,), dtype=tl.int32), (n,))
    
    # Copy A to L and U
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = tl.load(A_batch + i * stride_A_n + j * stride_A_k)
            else:
                L[i, j] = tl.load(A_batch + i * stride_A_n + j * stride_A_k)
    
    # Initialize P
    for i in range(n):
        P[i] = i
    
    # LU decomposition with partial pivoting
    for i in range(n):
        if pivot:
            # Find the pivot
            max_val = tl.abs(U[i, i])
            max_idx = i
            for j in range(i + 1, n):
                if tl.abs(U[j, i]) > max_val:
                    max_val = tl.abs(U[j, i])
                    max_idx = j
            
            # Swap rows in U
            if max_idx != i:
                for j in range(n):
                    temp = U[i, j]
                    U[i, j] = U[max_idx, j]
                    U[max_idx, j] = temp
                
                # Swap rows in L
                for j in range(i):
                    temp = L[i, j]
                    L[i, j] = L[max_idx, j]
                    L[max_idx, j] = temp
                
                # Update permutation
                temp = P[i]
                P[i] = P[max_idx]
                P[max_idx] = temp
        
        # Forward elimination
        for j in range(i + 1, n):
            if tl.abs(U[i, i]) > 1e-12:
                L[j, i] = U[j, i] / U[i, i]
                for k in range(i + 1, n):
                    U[j, k] = U[j, k] - L[j, i] * U[i, k]
    
    # Forward substitution to solve L * Y = P * B
    Y = tl.shared_ptr(tl.zeros((n, k), dtype=tl.float32), (n, k))
    for i in range(n):
        for j in range(k):
            Y[i, j] = tl.load(B_batch + P[i] * stride_B_n + j * stride_B_k)
    
    for i in range(n):
        for j in range(k):
            for l in range(i):
                Y[i, j] -= L[i, l] * Y[l, j]
    
    # Backward substitution to solve U * X = Y
    for i in range(n - 1, -1, -1):
        for j in range(k):
            for l in range(i + 1, n):
                Y[i, j] -= U[i, l] * tl.load(X_batch + l * stride_X_n + j * stride_X_k)
            if tl.abs(U[i, i]) > 1e-12:
                tl.store(X_batch + i * stride_X_n + j * stride_X_k, Y[i, j] / U[i, i])
            else:
                tl.store(X_batch + i * stride_X_n + j * stride_X_k, 0.0)

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")
    if A.shape[:-2] != Bs.shape[:-2]:
        raise ValueError("Batch dimensions of A and Bs must match")
    if A.shape[-1] != Bs.shape[-2]:
        raise ValueError("Last dimension of A must match second to last dimension of Bs")
    
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    k = Bs.shape[-1]
    
    if out is None:
        out = torch.empty_like(Bs)
    else:
        if out.shape != Bs.shape:
            raise ValueError("Output tensor must have the same shape as Bs")
    
    # Flatten batch dimensions
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Launch kernel
    grid = (batch_size,)
    block = (1,)
    
    # Use a simple kernel for small matrices or fallback to PyTorch for large ones
    if n <= 128:
        _solve_multiple_lu_kernel[grid, block](
            A, Bs, out,
            A.stride(-3) if len(A.shape) > 2 else 0,
            A.stride(-2),
            A.stride(-1),
            Bs.stride(-3) if len(Bs.shape) > 2 else 0,
            Bs.stride(-2),
            Bs.stride(-1),
            out.stride(-3) if len(out.shape) > 2 else 0,
            out.stride(-2),
            out.stride(-1),
            batch_size, n, k,
            pivot,
            BLOCK_SIZE=32
        )
    else:
        # For larger matrices, use PyTorch's built-in solver
        A_flat = A.view(-1, n, n)
        Bs_flat = Bs.view(-1, n, k)
        out_flat = out.view(-1, n, k)
        
        for i in range(A_flat.shape[0]):
            if pivot:
                lu, pivots = torch.lu(A_flat[i])
                out_flat[i] = torch.lu_solve(Bs_flat[i], lu, pivots)
            else:
                out_flat[i] = torch.solve(Bs_flat[i], A_flat[i]).solution
    
    return out

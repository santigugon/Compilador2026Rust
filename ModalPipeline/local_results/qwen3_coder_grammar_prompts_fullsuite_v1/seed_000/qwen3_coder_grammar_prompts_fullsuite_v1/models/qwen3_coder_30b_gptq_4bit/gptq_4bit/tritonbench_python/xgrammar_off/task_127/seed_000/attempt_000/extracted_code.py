import torch
import triton
import triton.language as tl
import math

@triton.jit
def _det_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Each block handles one batch element
    batch_id = tl.program_id(0)
    
    # Load matrix A for this batch
    A_block = tl.block_ptr(A_ptr, (batch_size, n, n), (n * n, n, 1), (batch_id, 0, 0), (1, n, n))
    
    # Allocate workspace for LU decomposition
    L = tl.zeros((n, n), dtype=tl.float32)
    U = tl.zeros((n, n), dtype=tl.float32)
    
    # Copy input matrix to U
    for i in range(n):
        for j in range(n):
            U[i, j] = tl.load(A_block, (i, j))
    
    # Perform LU decomposition
    for k in range(n):
        # Find pivot
        pivot = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[pivot, k]):
                pivot = i
        
        # Swap rows if needed
        if pivot != k:
            for j in range(n):
                temp = U[k, j]
                U[k, j] = U[pivot, j]
                U[pivot, j] = temp
        
        # Check for zero pivot
        if abs(U[k, k]) < 1e-10:
            # Matrix is singular, return 0
            tl.store(out_ptr + batch_id, 0.0)
            return
        
        # Eliminate
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] = U[i, j] - factor * U[k, j]
    
    # Compute determinant as product of diagonal elements
    det = 1.0
    for i in range(n):
        det *= U[i, i]
    
    # Store result
    tl.store(out_ptr + batch_id, det)

def det(A, *, out=None):
    # Handle scalar case
    if A.dim() == 2:
        batch_dims = ()
        n = A.shape[-1]
    else:
        batch_dims = A.shape[:-2]
        n = A.shape[-1]
    
    # Check if matrix is square
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input matrix must be square")
    
    # Compute batch size
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
    else:
        if out.shape != batch_dims:
            raise ValueError("Output tensor shape does not match batch dimensions")
    
    # Handle special cases
    if n == 1:
        # For 1x1 matrix, determinant is just the element
        if out is None:
            out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
        for i in range(batch_size):
            out[i] = A[i].item()
        return out
    
    if n == 2:
        # For 2x2 matrix: det = a*d - b*c
        if out is None:
            out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
        for i in range(batch_size):
            a = A[i, 0, 0]
            b = A[i, 0, 1]
            c = A[i, 1, 0]
            d = A[i, 1, 1]
            out[i] = a * d - b * c
        return out
    
    # For larger matrices, use the general algorithm
    block = 256
    grid = (batch_size,)
    
    # Allocate output tensor
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
    
    # Launch kernel
    _det_kernel[grid](A, out, batch_size, n, BLOCK=block)
    
    return out

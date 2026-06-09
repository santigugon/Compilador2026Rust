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
        for i in range(k+1, n):
            if abs(U[i, k]) > abs(U[pivot, k]):
                pivot = i
        
        # Swap rows if needed
        if pivot != k:
            for j in range(n):
                temp = U[k, j]
                U[k, j] = U[pivot, j]
                U[pivot, j] = temp
        
        # Check for zero pivot
        if abs(U[k, k]) < 1e-12:
            # Matrix is singular, return 0
            tl.store(out_ptr + batch_id, 0.0)
            return
        
        # Eliminate
        for i in range(k+1, n):
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
    # Handle scalar input
    if A.dim() == 0:
        return torch.tensor(A.item())
    
    # Get batch dimensions and matrix size
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    # Check if matrix is square
    if A.shape[-2] != n:
        raise ValueError("Input matrix must be square")
    
    # Handle batched matrices
    if len(batch_dims) > 0:
        # Flatten batch dimensions for processing
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        
        # Create output tensor
        if out is None:
            out = torch.empty(batch_size, dtype=A.dtype, device=A.device)
        else:
            if out.shape != (batch_size,):
                raise ValueError("Output tensor shape does not match batch size")
        
        # Launch kernel
        block = 256
        grid = (batch_size,)
        _det_kernel[grid](A, out, batch_size, n, BLOCK=block)
        
        # Reshape output to match batch dimensions
        if len(batch_dims) > 1:
            out = out.view(batch_dims)
        return out
    else:
        # Single matrix case
        if out is None:
            out = torch.empty(1, dtype=A.dtype, device=A.device)
        else:
            if out.shape != (1,):
                raise ValueError("Output tensor shape does not match expected size")
        
        # Launch kernel for single matrix
        block = 256
        grid = (1,)
        _det_kernel[grid](A, out, 1, n, BLOCK=block)
        return out[0]

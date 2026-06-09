import torch
import triton
import triton.language as tl

def _det_kernel(A_ptr, out_ptr, batch_size, n, stride_batch_A, stride_row_A, stride_col_A, stride_batch_out, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Load matrix A for this batch
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr + batch_idx * stride_batch_A,
        shape=(n, n),
        strides=(stride_row_A, stride_col_A),
        offsets=(0, 0),
        block_shape=(BLOCK, BLOCK),
        order=(0, 1)
    )
    A = tl.load(A_block_ptr, boundary_check=(0, 1))
    
    # Compute determinant using LU decomposition
    # For simplicity, we'll use a basic approach with partial pivoting
    # This is a simplified version and may not be numerically stable for all cases
    
    # Initialize determinant
    det = tl.full([], 1.0, dtype=tl.float64)
    
    # Create a copy of the matrix for LU decomposition
    L = tl.zeros((n, n), dtype=tl.float64)
    U = tl.zeros((n, n), dtype=tl.float64)
    
    # Copy A to U
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j]
            else:
                L[i, j] = A[i, j]
    
    # Perform LU decomposition
    for k in range(n):
        # Find pivot
        pivot = k
        for i in range(k+1, n):
            if tl.abs(U[i, k]) > tl.abs(U[pivot, k]):
                pivot = i
        
        # Swap rows if needed
        if pivot != k:
            for j in range(n):
                U[k, j], U[pivot, j] = U[pivot, j], U[k, j]
                if j < k:
                    L[k, j], L[pivot, j] = L[pivot, j], L[k, j]
            
        # Check for zero pivot
        if tl.abs(U[k, k]) < 1e-12:
            det = tl.full([], 0.0, dtype=tl.float64)
            break
        
        # Update L and U
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k+1, n):
                U[i, j] = U[i, j] - L[i, k] * U[k, j]
    
    # Compute determinant as product of diagonal elements of U
    for i in range(n):
        det = det * U[i, i]
    
    # Store result
    out = tl.make_block_ptr(
        base=out_ptr + batch_idx * stride_batch_out,
        shape=(),
        strides=(),
        offsets=(),
        block_shape=(),
        order=()
    )
    tl.store(out_ptr + batch_idx * stride_batch_out, det)


def linalg_det(A, *, out=None):
    if A.dim() < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input must be square matrices")
    
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
    else:
        if out.shape != batch_dims:
            raise ValueError("Output tensor must have the same batch dimensions as input")
        if out.dtype != torch.float64:
            raise ValueError("Output tensor must be of type float64")
    
    # Handle scalar case
    if len(batch_dims) == 0:
        batch_size = 1
        batch_dims = (1,)
    else:
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
    
    # For small matrices, we can use a direct approach
    if n <= 4:
        # Use torch implementation for small matrices
        return torch.linalg.det(A)
    
    # For larger matrices, use Triton kernel
    BLOCK = 32
    grid = (batch_size,)
    
    # Get strides
    stride_batch_A = A.stride(-3) if len(A.shape) >= 3 else 0
    stride_row_A = A.stride(-2) if len(A.shape) >= 2 else 0
    stride_col_A = A.stride(-1) if len(A.shape) >= 1 else 0
    stride_batch_out = out.stride(0) if len(out.shape) >= 1 else 0
    
    _det_kernel[grid](A, out, batch_size, n, stride_batch_A, stride_row_A, stride_col_A, stride_batch_out, BLOCK=BLOCK)
    return out
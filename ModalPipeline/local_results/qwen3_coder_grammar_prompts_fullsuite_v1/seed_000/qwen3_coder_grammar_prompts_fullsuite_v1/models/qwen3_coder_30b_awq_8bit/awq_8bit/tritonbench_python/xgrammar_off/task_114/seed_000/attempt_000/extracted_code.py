import torch
import triton
import triton.language as tl

@triton.jit
def _determinant_lu_kernel(
    A_ptr, 
    out_ptr, 
    n, 
    batch_size,
    pivot: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    A_batch = A_ptr + batch_idx * n * n
    out_batch = out_ptr + batch_idx
    
    # Initialize determinant
    det = tl.full([1], 1.0, dtype=tl.float64)
    sign = tl.full([1], 1.0, dtype=tl.float64)
    
    # Copy matrix to work with
    A_work = tl.full([BLOCK_SIZE, BLOCK_SIZE], 0.0, dtype=tl.float64)
    
    # Load matrix
    for i in range(n):
        for j in range(n):
            A_work[i, j] = tl.load(A_batch + i * n + j)
    
    # LU decomposition with partial pivoting
    for k in range(n):
        # Find pivot
        if pivot:
            max_val = tl.abs(A_work[k, k])
            pivot_row = k
            for i in range(k + 1, n):
                if tl.abs(A_work[i, k]) > max_val:
                    max_val = tl.abs(A_work[i, k])
                    pivot_row = i
            
            # Swap rows if needed
            if pivot_row != k:
                sign = -sign
                for j in range(n):
                    temp = A_work[k, j]
                    A_work[k, j] = A_work[pivot_row, j]
                    A_work[pivot_row, j] = temp
        
        # Check for zero pivot
        if tl.abs(A_work[k, k]) < 1e-12:
            det = 0.0
            break
        
        # Update determinant
        det = det * A_work[k, k]
        
        # Eliminate column
        for i in range(k + 1, n):
            factor = A_work[i, k] / A_work[k, k]
            for j in range(k + 1, n):
                A_work[i, j] = A_work[i, j] - factor * A_work[k, j]
    
    # Store result
    tl.store(out_batch, det * sign)

def determinant_lu(A, *, pivot=True, out=None):
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input tensor must represent square matrices")
    
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
    
    if out.shape != batch_dims:
        raise ValueError("Output tensor shape does not match batch dimensions")
    
    # Launch kernel
    grid = (batch_size, 1, 1)
    BLOCK_SIZE = 32
    
    _determinant_lu_kernel[grid](
        A,
        out,
        n,
        batch_size,
        pivot,
        BLOCK_SIZE
    )
    
    return out

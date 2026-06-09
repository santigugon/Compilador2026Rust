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
    
    # Shared memory for the matrix
    A_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize determinant
    det = tl.full([], 1.0, dtype=tl.float32)
    sign = tl.full([], 1.0, dtype=tl.float32)
    
    # Copy matrix to shared memory
    for i in range(0, n, BLOCK_SIZE):
        for j in range(0, n, BLOCK_SIZE):
            if i + tl.arange(0, BLOCK_SIZE) < n and j + tl.arange(0, BLOCK_SIZE) < n:
                row = i + tl.arange(0, BLOCK_SIZE)
                col = j + tl.arange(0, BLOCK_SIZE)
                for k in range(BLOCK_SIZE):
                    for l in range(BLOCK_SIZE):
                        if row[k] < n and col[l] < n:
                            A_shared[k, l] = tl.load(A_batch + (row[k] * n + col[l]))
    
    # LU decomposition with partial pivoting
    for k in range(n):
        # Find pivot
        if pivot:
            max_val = tl.abs(A_shared[k, k])
            pivot_row = k
            for i in range(k + 1, n):
                if tl.abs(A_shared[i, k]) > max_val:
                    max_val = tl.abs(A_shared[i, k])
                    pivot_row = i
            
            # Swap rows if needed
            if pivot_row != k:
                sign = -sign
                for j in range(n):
                    temp = A_shared[k, j]
                    A_shared[k, j] = A_shared[pivot_row, j]
                    A_shared[pivot_row, j] = temp
        
        # Check for zero pivot
        if A_shared[k, k] == 0.0:
            det = 0.0
            break
        
        # Update determinant
        det = det * A_shared[k, k]
        
        # Eliminate column
        for i in range(k + 1, n):
            if A_shared[k, k] != 0.0:
                factor = A_shared[i, k] / A_shared[k, k]
                for j in range(k + 1, n):
                    A_shared[i, j] = A_shared[i, j] - factor * A_shared[k, j]
    
    # Store result
    tl.store(out_batch, det * sign)

def determinant_lu(A, *, pivot=True, out=None):
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float32, device=A.device)
    
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Unsupported dtype")
    
    # Use appropriate block size
    BLOCK_SIZE = min(32, n)
    
    # Launch kernel
    grid = (batch_size, 1, 1)
    _determinant_lu_kernel[grid](
        A,
        out,
        n,
        batch_size,
        pivot,
        BLOCK_SIZE
    )
    
    return out

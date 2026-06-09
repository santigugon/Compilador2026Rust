import torch
import triton
import triton.language as tl

@triton.jit
def _determinant_lu_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, pivot: tl.constexpr, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    
    # Load matrix A for this batch
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr + batch_idx * n * n,
        shape=(n, n),
        strides=(n, 1),
        offsets=(0, 0),
        block_shape=(BLOCK, BLOCK),
        order=(1, 0)
    )
    
    # Initialize output with 1.0
    det = tl.full([], 1.0, dtype=tl.float64)
    sign = tl.full([], 1.0, dtype=tl.float64)
    
    # Copy matrix A to local memory for LU decomposition
    A_local = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # LU decomposition with partial pivoting
    for k in range(n):
        # Find pivot
        if pivot:
            # Find maximum element in column k
            max_val = tl.abs(A_local[k, k])
            pivot_row = k
            for i in range(k + 1, n):
                if tl.abs(A_local[i, k]) > max_val:
                    max_val = tl.abs(A_local[i, k])
                    pivot_row = i
            
            # Swap rows if needed
            if pivot_row != k:
                # Swap rows k and pivot_row
                for j in range(n):
                    temp = A_local[k, j]
                    A_local[k, j] = A_local[pivot_row, j]
                    A_local[pivot_row, j] = temp
                sign = -sign
            
            # Check for zero pivot
            if A_local[k, k] == 0.0:
                det = 0.0
                break
        
        # Update determinant with diagonal element
        det = det * A_local[k, k]
        
        # Perform elimination
        if k < n - 1:
            # Compute multipliers
            for i in range(k + 1, n):
                if A_local[k, k] != 0.0:
                    A_local[i, k] = A_local[i, k] / A_local[k, k]
                    # Update remaining elements
                    for j in range(k + 1, n):
                        A_local[i, j] = A_local[i, j] - A_local[i, k] * A_local[k, j]
    
    # Store result
    tl.store(out_ptr + batch_idx, det * sign)

def determinant_lu(A, *, pivot=True, out=None):
    # Validate input
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input tensor must be square matrices")
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
    else:
        if out.shape != batch_dims:
            raise ValueError("Output tensor must have shape (*,)")
    
    # Handle scalar case
    if len(batch_dims) == 0:
        batch_size = 1
        batch_dims = (1,)
    else:
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
    
    # Launch kernel
    block = 16
    grid = (batch_size,)
    
    # For small matrices, we can use a simpler approach
    if n <= 32:
        # Use a more direct approach for small matrices
        if pivot:
            # For small matrices, we can compute directly
            if batch_size == 1:
                # Single matrix case
                if n == 1:
                    out.fill_(A[0, 0])
                elif n == 2:
                    out.fill_(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
                elif n == 3:
                    out.fill_(A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) - 
                             A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + 
                             A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))
                else:
                    # Fall back to full implementation for larger matrices
                    _determinant_lu_kernel[grid](A, out, batch_size, n, pivot, BLOCK=block)
            else:
                # Batch case
                _determinant_lu_kernel[grid](A, out, batch_size, n, pivot, BLOCK=block)
        else:
            # No pivoting case
            _determinant_lu_kernel[grid](A, out, batch_size, n, pivot, BLOCK=block)
    else:
        # For larger matrices, use the full kernel
        _determinant_lu_kernel[grid](A, out, batch_size, n, pivot, BLOCK=block)
    
    return out

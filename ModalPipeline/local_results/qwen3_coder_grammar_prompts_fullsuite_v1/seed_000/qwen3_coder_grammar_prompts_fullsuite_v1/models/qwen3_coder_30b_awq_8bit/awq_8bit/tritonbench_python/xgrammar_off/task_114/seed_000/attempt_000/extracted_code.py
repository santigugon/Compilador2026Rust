import torch
import triton
import triton.language as tl
import math

@triton.jit
def _determinant_lu_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, pivot: tl.constexpr, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Each batch processes one matrix
    matrix_offset = batch_idx * n * n
    A_base = A_ptr + matrix_offset
    
    # Create a copy of the matrix for LU decomposition
    L = tl.full((n, n), 0.0, dtype=tl.float32)
    U = tl.full((n, n), 0.0, dtype=tl.float32)
    
    # Initialize U with A values
    for i in range(n):
        for j in range(n):
            offset = i * n + j
            U[i, j] = tl.load(A_base + offset)
    
    # Initialize L as identity matrix
    for i in range(n):
        L[i, i] = 1.0
    
    # Permutation sign tracking
    sign = 1.0
    
    # LU decomposition with partial pivoting
    for k in range(n):
        # Find pivot
        if pivot:
            max_val = tl.abs(U[k, k])
            pivot_row = k
            for i in range(k + 1, n):
                if tl.abs(U[i, k]) > max_val:
                    max_val = tl.abs(U[i, k])
                    pivot_row = i
            
            # Swap rows in U and L if needed
            if pivot_row != k:
                sign = -sign
                for j in range(n):
                    temp = U[k, j]
                    U[k, j] = U[pivot_row, j]
                    U[pivot_row, j] = temp
                    
                    temp = L[k, j]
                    L[k, j] = L[pivot_row, j]
                    L[pivot_row, j] = temp
        
        # Check for zero pivot
        if abs(U[k, k]) < 1e-12:
            # Return zero determinant if singular
            tl.store(out_ptr + batch_idx, 0.0)
            return
        
        # Compute L and U
        for i in range(k + 1, n):
            if abs(U[k, k]) > 1e-12:
                L[i, k] = U[i, k] / U[k, k]
                for j in range(k + 1, n):
                    U[i, j] = U[i, j] - L[i, k] * U[k, j]
    
    # Compute determinant as product of diagonal elements of U
    det = sign
    for i in range(n):
        det = det * U[i, i]
    
    tl.store(out_ptr + batch_idx, det)

def determinant_lu(A, *, pivot=True, out=None):
    # Handle scalar input
    if A.dim() == 0:
        return A.clone()
    
    # Handle 1D input (vector)
    if A.dim() == 1:
        if A.shape[0] == 1:
            return A[0].clone()
        else:
            raise ValueError("Input must be a square matrix")
    
    # Handle 2D input (single matrix)
    if A.dim() == 2:
        batch_shape = ()
        n = A.shape[-1]
        if A.shape[-2] != n:
            raise ValueError("Input must be a square matrix")
        batch_size = 1
    else:
        # Handle batched input
        batch_shape = A.shape[:-2]
        n = A.shape[-1]
        if A.shape[-2] != n:
            raise ValueError("Input must be a square matrix")
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    else:
        if out.shape != batch_shape:
            raise ValueError("Output tensor shape does not match input batch shape")
        if out.dtype != A.dtype:
            raise ValueError("Output tensor dtype does not match input dtype")
        if out.device != A.device:
            raise ValueError("Output tensor device does not match input device")
    
    # For small matrices, use PyTorch's implementation for better numerical stability
    if n <= 4:
        if out is None:
            out = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
        if batch_size == 1:
            out.fill_(torch.det(A))
        else:
            for i in range(batch_size):
                batch_A = A[i] if batch_size > 1 else A
                out[i] = torch.det(batch_A)
        return out
    
    # For larger matrices, use Triton implementation
    if batch_size > 0:
        block = 256
        grid = (triton.cdiv(batch_size, block),)
        _determinant_lu_kernel[grid](A, out, batch_size, n, pivot, BLOCK=block)
    
    return out

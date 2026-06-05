import torch
import triton
import triton.language as tl
import math

@triton.jit
def _det_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, stride_batch: tl.constexpr, stride_row: tl.constexpr, stride_col: tl.constexpr, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    
    # Load matrix A for this batch
    A_block = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for i in range(n):
        for j in range(n):
            if i < n and j < n:
                offset = batch_idx * stride_batch + i * stride_row + j * stride_col
                A_block[i, j] = tl.load(A_ptr + offset, mask=(i < n) & (j < n), other=0.0)
    
    # Compute determinant using LU decomposition
    det = 1.0
    # Create a copy of the matrix for LU decomposition
    L = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    U = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # Initialize U with A
    for i in range(n):
        for j in range(n):
            if i < n and j < n:
                U[i, j] = A_block[i, j]
    
    # LU decomposition with partial pivoting
    for k in range(n):
        # Find pivot
        pivot_row = k
        pivot_val = tl.abs(U[k, k])
        for i in range(k + 1, n):
            if tl.abs(U[i, k]) > pivot_val:
                pivot_val = tl.abs(U[i, k])
                pivot_row = i
        
        # Swap rows if needed
        if pivot_row != k:
            for j in range(n):
                temp = U[k, j]
                U[k, j] = U[pivot_row, j]
                U[pivot_row, j] = temp
            det = -det
        
        # Check for singular matrix
        if tl.abs(U[k, k]) < 1e-12:
            det = 0.0
            break
        
        # Update determinant
        det *= U[k, k]
        
        # Eliminate column
        for i in range(k + 1, n):
            if tl.abs(U[k, k]) > 1e-12:
                factor = U[i, k] / U[k, k]
                for j in range(k, n):
                    U[i, j] = U[i, j] - factor * U[k, j]
    
    # Store result
    tl.store(out_ptr + batch_idx, det)

def linalg_det(A, *, out=None):
    if A.dim() < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input must be square matrices")
    
    # Handle batch dimensions
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    # Create output tensor
    if out is not None:
        if out.shape != batch_dims:
            raise ValueError("Output tensor shape does not match batch dimensions")
        out = out
    else:
        out = torch.empty(batch_dims, dtype=torch.float32, device=A.device)
    
    # Handle scalar case
    if len(batch_dims) == 0:
        batch_size = 1
        batch_dims = (1,)
    else:
        batch_size = math.prod(batch_dims)
    
    # Launch kernel
    block = 32
    grid = (batch_size,)
    
    # Get strides
    stride_batch = A.stride(-3) if len(A.shape) >= 3 else 1
    stride_row = A.stride(-2) if len(A.shape) >= 2 else 1
    stride_col = A.stride(-1) if len(A.shape) >= 1 else 1
    
    # For small matrices, we can use a simpler approach
    if n <= 32:
        _det_kernel[grid](
            A, out, batch_size, n, stride_batch, stride_row, stride_col, BLOCK=block
        )
    else:
        # For larger matrices, fall back to PyTorch for numerical stability
        # This is a simplified approach - in practice, you'd want a more robust implementation
        if out is not None:
            out.copy_(torch.det(A))
        else:
            out = torch.det(A)
    
    return out

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _det_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, stride_batch: tl.constexpr, stride_row: tl.constexpr, stride_col: tl.constexpr, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    
    # Load matrix into shared memory
    A_shared = tl.shared_ptr(A_ptr, shape=(BLOCK, BLOCK), dtype=tl.float32)
    
    # Initialize determinant
    det = tl.full([], 1.0, dtype=tl.float32)
    
    # Copy matrix to shared memory
    for i in range(n):
        for j in range(n):
            if i < BLOCK and j < BLOCK:
                A_shared[i, j] = tl.load(A_ptr + batch_idx * stride_batch + i * stride_row + j * stride_col)
    
    # Gaussian elimination with partial pivoting
    for k in range(n):
        # Find pivot
        pivot_val = tl.abs(A_shared[k, k])
        pivot_row = k
        for i in range(k + 1, n):
            if tl.abs(A_shared[i, k]) > pivot_val:
                pivot_val = tl.abs(A_shared[i, k])
                pivot_row = i
        
        # Swap rows if needed
        if pivot_row != k:
            for j in range(n):
                temp = A_shared[k, j]
                A_shared[k, j] = A_shared[pivot_row, j]
                A_shared[pivot_row, j] = temp
            det = -det
        
        # Check for zero pivot
        if tl.abs(A_shared[k, k]) < 1e-12:
            det = tl.zeros([], dtype=tl.float32)
            break
        
        # Update determinant
        det = det * A_shared[k, k]
        
        # Eliminate column
        for i in range(k + 1, n):
            factor = A_shared[i, k] / A_shared[k, k]
            for j in range(k + 1, n):
                A_shared[i, j] = A_shared[i, j] - factor * A_shared[k, j]
    
    # Store result
    tl.store(out_ptr + batch_idx, det)

def _det_batched(A, out=None):
    if A.dim() < 2:
        raise ValueError("Input must be at least 2D")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input must be square matrices")
    
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float32, device=A.device)
    else:
        if out.shape != batch_dims:
            raise ValueError("Output shape mismatch")
    
    # Handle scalar case
    if len(batch_dims) == 0:
        batch_size = 1
        batch_dims = (1,)
    else:
        batch_size = math.prod(batch_dims)
    
    # For small matrices, use direct computation
    if n <= 3:
        return _det_small_matrices(A, out)
    
    # For larger matrices, use Gaussian elimination
    block = 32
    grid = (batch_size,)
    
    # Flatten batch dimensions for kernel
    A_flat = A.view(batch_size, n, n)
    out_flat = out.view(batch_size)
    
    # Compute strides
    stride_batch = n * n
    stride_row = n
    stride_col = 1
    
    _det_kernel[grid](
        A_flat, out_flat, batch_size, n, stride_batch, stride_row, stride_col, BLOCK=block
    )
    
    return out

def _det_small_matrices(A, out):
    """Direct computation for small matrices"""
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    
    if n == 1:
        out.copy_(A[..., 0, 0])
    elif n == 2:
        out.copy_(A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0])
    elif n == 3:
        out.copy_(
            A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1]) -
            A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0]) +
            A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
        )
    else:
        # Fall back to torch for larger matrices
        out.copy_(torch.det(A))
    
    return out

def linalg_det(A, *, out=None):
    # Handle scalar input
    if A.dim() == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=A.device)
    
    # Handle batched input
    if A.dim() < 2:
        raise ValueError("Input must be at least 2D")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input must be square matrices")
    
    # Determine output dtype
    if A.dtype in (torch.complex64, torch.complex128):
        out_dtype = torch.complex64 if A.dtype == torch.complex64 else torch.complex128
    else:
        out_dtype = torch.float32 if A.dtype in (torch.float32, torch.int32) else torch.float64
    
    if out is None:
        out = torch.empty(batch_dims, dtype=out_dtype, device=A.device)
    else:
        if out.shape != batch_dims:
            raise ValueError("Output shape mismatch")
        if out.dtype != out_dtype:
            raise ValueError("Output dtype mismatch")
    
    # For small matrices, use direct computation
    if n <= 3:
        return _det_small_matrices(A, out)
    
    # For larger matrices, use Gaussian elimination
    return _det_batched(A, out)

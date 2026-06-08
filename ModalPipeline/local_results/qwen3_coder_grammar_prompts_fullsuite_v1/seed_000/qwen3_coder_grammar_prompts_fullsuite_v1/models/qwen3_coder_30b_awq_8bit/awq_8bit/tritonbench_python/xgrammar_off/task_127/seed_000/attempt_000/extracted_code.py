import torch
import triton
import triton.language as tl
import math

@triton.jit
def _det_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, stride_batch: tl.constexpr, stride_row: tl.constexpr, stride_col: tl.constexpr, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Load matrix A for this batch
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr + batch_idx * stride_batch,
        shape=(n, n),
        strides=(stride_row, stride_col),
        offsets=(0, 0),
        block_shape=(BLOCK, BLOCK),
        order=(0, 1)
    )
    
    # Load matrix into shared memory
    A = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Initialize determinant
    det = tl.full((), 1.0, dtype=tl.float64)
    
    # Gaussian elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_val = tl.abs(A[k, k])
        pivot_row = k
        for i in range(k + 1, n):
            if tl.abs(A[i, k]) > max_val:
                max_val = tl.abs(A[i, k])
                pivot_row = i
        
        # Swap rows if needed
        if pivot_row != k:
            for j in range(n):
                temp = A[k, j]
                A[k, j] = A[pivot_row, j]
                A[pivot_row, j] = temp
            det = -det
        
        # Check for zero pivot
        if tl.abs(A[k, k]) < 1e-12:
            det = tl.zeros((), dtype=tl.float64)
            break
        
        # Update determinant
        det = det * A[k, k]
        
        # Eliminate column
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - factor * A[k, j]
    
    # Store result
    tl.store(out_ptr + batch_idx, det)

def linalg_det(A, *, out=None):
    if A.dim() < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input must be square matrices")
    
    # Handle batch dimensions
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float64, device=A.device)
    else:
        if out.shape != batch_dims:
            raise ValueError("Output tensor must have the same batch dimensions as input")
        if out.dtype != torch.float64:
            raise ValueError("Output tensor must be of type float64")
    
    # For small matrices, use direct computation
    if n <= 2:
        if n == 1:
            out = A[..., 0, 0].to(torch.float64)
        elif n == 2:
            out = (A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]).to(torch.float64)
        return out
    
    # For larger matrices, use Triton kernel
    block = 16
    grid = (batch_size,)
    
    # Compute strides for batched operations
    stride_batch = A.stride(-3) if len(A.shape) >= 3 else 1
    stride_row = A.stride(-2) if len(A.shape) >= 2 else 1
    stride_col = A.stride(-1) if len(A.shape) >= 1 else 1
    
    _det_kernel[grid](
        A, out, batch_size, n, stride_batch, stride_row, stride_col, BLOCK=block
    )
    
    return out

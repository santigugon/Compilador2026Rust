import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _solve_kernel(
    A_ptr, B_ptr, out_ptr,
    n, batch_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    LEFT: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Compute offset for batch
    A_batch_offset = batch_idx * n * n
    B_batch_offset = batch_idx * n
    out_batch_offset = batch_idx * n
    
    # Load matrix A and B
    A = tl.zeros((n, n), dtype=tl.float64 if IS_DOUBLE else tl.float32)
    B = tl.zeros((n,), dtype=tl.float64 if IS_DOUBLE else tl.float32)
    
    # Load B
    for i in range(n):
        B[i] = tl.load(B_ptr + B_batch_offset + i)
    
    # Load A
    for i in range(n):
        for j in range(n):
            A[i, j] = tl.load(A_ptr + A_batch_offset + i * n + j)
    
    # Solve Ax = B or xA = B
    if LEFT:
        # Solve Ax = B
        x = tl.zeros((n,), dtype=tl.float64 if IS_DOUBLE else tl.float32)
        for i in range(n):
            x[i] = B[i]
        
        # Forward elimination
        for i in range(n):
            if A[i, i] == 0.0:
                # Handle zero pivot
                for k in range(i+1, n):
                    if A[k, i] != 0.0:
                        # Swap rows
                        for j in range(n):
                            A[i, j], A[k, j] = A[k, j], A[i, j]
                        x[i], x[k] = x[k], x[i]
                        break
            if A[i, i] != 0.0:
                for j in range(i+1, n):
                    factor = A[j, i] / A[i, i]
                    for k in range(i+1, n):
                        A[j, k] -= factor * A[i, k]
                    x[j] -= factor * x[i]
        
        # Back substitution
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                x[i] -= A[i, j] * x[j]
            x[i] /= A[i, i]
    else:
        # Solve xA = B
        x = tl.zeros((n,), dtype=tl.float64 if IS_DOUBLE else tl.float32)
        for i in range(n):
            x[i] = B[i]
        
        # Forward elimination
        for i in range(n):
            if A[i, i] == 0.0:
                # Handle zero pivot
                for k in range(i+1, n):
                    if A[i, k] != 0.0:
                        # Swap columns
                        for j in range(n):
                            A[j, i], A[j, k] = A[j, k], A[j, i]
                        x[i], x[k] = x[k], x[i]
                        break
            if A[i, i] != 0.0:
                for j in range(i+1, n):
                    factor = A[i, j] / A[i, i]
                    for k in range(i+1, n):
                        A[k, j] -= factor * A[k, i]
                    x[j] -= factor * x[i]
        
        # Back substitution
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                x[i] -= A[j, i] * x[j]
            x[i] /= A[i, i]
    
    # Write result
    for i in range(n):
        tl.store(out_ptr + out_batch_offset + i, x[i])

def solve(A: torch.Tensor, B: torch.Tensor, *, left: bool = True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if A.dim() < 2 or B.dim() < 1:
        raise ValueError("A must be at least 2D and B at least 1D")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be square")
    
    if A.shape[-1] != B.shape[-1]:
        raise ValueError("Last dimension of A must match last dimension of B")
    
    batch_dims_A = A.shape[:-2]
    batch_dims_B = B.shape[:-1]
    
    if batch_dims_A != batch_dims_B:
        raise ValueError("Batch dimensions of A and B must match")
    
    batch_size = 1
    for dim in batch_dims_A:
        batch_size *= dim
    
    n = A.shape[-1]
    
    if out is None:
        out = torch.empty_like(B)
    
    BLOCK_SIZE = 32
    
    # Determine if we're working with doubles or floats
    IS_DOUBLE = A.dtype in [torch.float64, torch.complex128]
    
    # Launch kernel
    grid = (batch_size,)
    _solve_kernel[grid](
        A.data_ptr(), B.data_ptr(), out.data_ptr(),
        n, batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_DOUBLE=IS_DOUBLE,
        LEFT=left
    )
    
    return out

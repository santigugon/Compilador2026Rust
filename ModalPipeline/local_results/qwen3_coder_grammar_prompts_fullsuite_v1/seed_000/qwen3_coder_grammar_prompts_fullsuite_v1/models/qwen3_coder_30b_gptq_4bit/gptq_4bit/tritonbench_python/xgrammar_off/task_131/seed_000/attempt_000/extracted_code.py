import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def _cholesky_solve_kernel(
    B_ptr, L_ptr, out_ptr,
    batch_size, n, k,
    upper: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Load B and L for this batch
    B_batch = B_ptr + batch_idx * n * k
    L_batch = L_ptr + batch_idx * n * n
    out_batch = out_ptr + batch_idx * n * k
    
    # Process each column of B
    for col in range(k):
        # Copy B to out
        for i in range(n):
            out_i = out_batch + i * k + col
            b_i = B_batch + i * k + col
            tl.store(out_i, tl.load(b_i))
        
        # Forward substitution or back substitution
        if upper:
            # Back substitution for upper triangular
            for i in range(n-1, -1, -1):
                out_i = out_batch + i * k + col
                out_val = tl.load(out_i)
                for j in range(i+1, n):
                    l_ij = L_batch + i * n + j
                    out_val -= tl.load(l_ij) * tl.load(out_batch + j * k + col)
                tl.store(out_i, out_val)
        else:
            # Forward substitution for lower triangular
            for i in range(n):
                out_i = out_batch + i * k + col
                out_val = tl.load(out_i)
                for j in range(i):
                    l_ji = L_batch + j * n + i
                    out_val -= tl.load(l_ji) * tl.load(out_batch + j * k + col)
                tl.store(out_i, out_val)

def cholesky_solve(B: torch.Tensor, L: torch.Tensor, upper: bool = False, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Validate inputs
    assert B.dim() >= 2, "B must have at least 2 dimensions"
    assert L.dim() >= 2, "L must have at least 2 dimensions"
    assert B.shape[-1] == L.shape[-1], "Last dimension of B and L must match"
    assert L.shape[-1] == L.shape[-2], "L must be square"
    
    # Handle batch dimensions
    batch_dims_B = B.shape[:-2]
    batch_dims_L = L.shape[:-2]
    assert batch_dims_B == batch_dims_L, "Batch dimensions of B and L must match"
    
    batch_size = 1
    for dim in batch_dims_B:
        batch_size *= dim
    
    n = L.shape[-1]
    k = B.shape[-1]
    
    # Determine output tensor
    if out is None:
        out = torch.empty_like(B)
    else:
        assert out.shape == B.shape, "Output tensor must have the same shape as B"
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (batch_size,)
    
    _cholesky_solve_kernel[grid](
        B, L, out,
        batch_size, n, k,
        upper=upper,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

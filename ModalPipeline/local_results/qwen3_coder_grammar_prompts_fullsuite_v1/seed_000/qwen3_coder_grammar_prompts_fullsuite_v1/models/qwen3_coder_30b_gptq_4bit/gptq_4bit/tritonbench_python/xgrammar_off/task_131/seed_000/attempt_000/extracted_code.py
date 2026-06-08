import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_solve_kernel(B_ptr, L_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, k: tl.constexpr, upper: tl.constexpr, BLOCK: tl.constexpr):
    # Compute the batch index
    batch_idx = tl.program_id(0)
    
    # Load B and L for this batch
    B_batch = B_ptr + batch_idx * n * k
    L_batch = L_ptr + batch_idx * n * n
    
    # Output tensor
    out_batch = out_ptr + batch_idx * n * k
    
    # Process each column of B
    for col in range(k):
        # Load the column of B
        b_col = tl.load(B_batch + col * n, mask=tl.arange(0, n) < n, other=0.0)
        
        # Forward substitution or back substitution depending on upper
        if upper:
            # Back substitution for upper triangular matrix
            for i in range(n - 1, -1, -1):
                # Compute the sum of known terms
                sum_val = tl.sum(b_col[tl.arange(0, n) > i] * tl.load(L_batch + i * n + tl.arange(0, n) > i, mask=tl.arange(0, n) > i, other=0.0))
                # Update the value
                b_col[i] = (b_col[i] - sum_val) / tl.load(L_batch + i * n + i, mask=True, other=0.0)
        else:
            # Forward substitution for lower triangular matrix
            for i in range(n):
                # Compute the sum of known terms
                sum_val = tl.sum(b_col[tl.arange(0, n) < i] * tl.load(L_batch + i * n + tl.arange(0, n) < i, mask=tl.arange(0, n) < i, other=0.0))
                # Update the value
                b_col[i] = (b_col[i] - sum_val) / tl.load(L_batch + i * n + i, mask=True, other=0.0)
        
        # Store the result
        tl.store(out_batch + col * n, b_col, mask=tl.arange(0, n) < n)

def cholesky_solve(B, L, upper=False, *, out=None):
    # Check dimensions
    batch_dims_B = B.shape[:-2]
    batch_dims_L = L.shape[:-2]
    
    # Ensure batch dimensions match
    if batch_dims_B != batch_dims_L:
        raise ValueError("Batch dimensions of B and L must match")
    
    # Get batch size
    batch_size = 1
    for dim in batch_dims_B:
        batch_size *= dim
    
    # Get matrix dimensions
    n = L.shape[-1]
    k = B.shape[-1]
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(B)
    else:
        if out.shape != B.shape:
            raise ValueError("Output tensor shape must match input tensor B")
    
    # Determine block size
    BLOCK = 256
    
    # Launch kernel
    grid = (batch_size,)
    _cholesky_solve_kernel[grid](
        B, L, out, 
        batch_size, n, k, 
        upper, 
        BLOCK=BLOCK
    )
    
    return out

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _determinant_lu_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, pivot: tl.constexpr, BLOCK: tl.constexpr):
    batch_id = tl.program_id(0)
    
    # Load matrix A for this batch
    A_block = tl.block_ptr(A_ptr, (batch_size, n, n), (0, 0, 0), (1, BLOCK, BLOCK), (0, 1, 2))
    
    # Initialize output
    det = tl.full([1], 1.0, dtype=tl.float64)
    sign = tl.full([1], 1.0, dtype=tl.float64)
    
    # Perform LU decomposition with optional pivoting
    for i in range(n):
        # Find pivot element
        if pivot:
            max_val = tl.abs(tl.load(A_block + (i, i)))
            max_idx = i
            for k in range(i + 1, n):
                val = tl.abs(tl.load(A_block + (k, i)))
                if val > max_val:
                    max_val = val
                    max_idx = k
            
            # Swap rows if needed
            if max_idx != i:
                sign = -sign
                for j in range(n):
                    temp = tl.load(A_block + (i, j))
                    tl.store(A_block + (i, j), tl.load(A_block + (max_idx, j)))
                    tl.store(A_block + (max_idx, j), temp)
        
        # Check for zero pivot
        pivot_val = tl.load(A_block + (i, i))
        if pivot_val == 0.0:
            det = 0.0
            break
        
        # Update determinant
        det = det * pivot_val
        
        # Perform elimination
        for j in range(i + 1, n):
            factor = tl.load(A_block + (j, i)) / tl.load(A_block + (i, i))
            for k in range(i + 1, n):
                current_val = tl.load(A_block + (j, k))
                new_val = current_val - factor * tl.load(A_block + (i, k))
                tl.store(A_block + (j, k), new_val)
    
    # Store result
    tl.store(out_ptr + batch_id, det * sign)

def determinant_lu(A, *, pivot=True, out=None):
    # Handle scalar input
    if A.dim() == 2:
        A = A.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = A.shape[0]
    
    n = A.shape[-1]
    
    # Validate input
    if A.shape[-2] != n:
        raise ValueError("Input matrix must be square")
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_size, dtype=torch.float64, device=A.device)
    else:
        if out.shape != (batch_size,):
            raise ValueError("Output tensor must have shape (batch_size,)")
    
    # Handle batch dimension
    if batch_size > 1:
        # Use a single kernel for all batches
        block = 32
        grid = (batch_size,)
        _determinant_lu_kernel[grid](A, out, batch_size, n, pivot, BLOCK=block)
    else:
        # For single batch, compute directly
        A_flat = A.view(-1, n, n)
        det = torch.det(A_flat)
        out.copy_(det)
    
    return out

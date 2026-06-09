import torch
import triton
import triton.language as tl

@triton.jit
def _ldl_factor_kernel(
    A_ptr, LD_ptr, pivots_ptr, 
    n, batch_size,
    stride_a_batch, stride_a_row, stride_a_col,
    stride_ld_batch, stride_ld_row, stride_ld_col,
    stride_piv_batch, stride_piv_row,
    hermitian: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    tid = tl.program_id(1)
    
    # Load matrix A for this batch
    A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i < n and j < n:
                a_idx = batch_idx * stride_a_batch + i * stride_a_row + j * stride_a_col
                A[i, j] = tl.load(A_ptr + a_idx)
    
    # Initialize LD and pivots
    LD = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    pivots = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # Copy A to LD
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i < n and j < n:
                ld_idx = batch_idx * stride_ld_batch + i * stride_ld_row + j * stride_ld_col
                tl.store(LD_ptr + ld_idx, A[i, j])
    
    # Perform LDL factorization
    for k in range(n):
        # Find pivot
        max_val = tl.abs(LD[k, k])
        pivot_idx = k
        for i in range(k + 1, n):
            if tl.abs(LD[i, k]) > max_val:
                max_val = tl.abs(LD[i, k])
                pivot_idx = i
        
        # Swap rows and columns if needed
        if pivot_idx != k:
            # Swap rows
            for j in range(n):
                temp = LD[k, j]
                LD[k, j] = LD[pivot_idx, j]
                LD[pivot_idx, j] = temp
            # Swap columns
            for i in range(n):
                temp = LD[i, k]
                LD[i, k] = LD[i, pivot_idx]
                LD[i, pivot_idx] = temp
            # Update pivot array
            pivots[k] = pivot_idx
        else:
            pivots[k] = k
            
        # Check for zero pivot
        if tl.abs(LD[k, k]) < 1e-12:
            # Set to zero for numerical stability
            for i in range(k, n):
                LD[i, k] = 0.0
            continue
            
        # Compute column k
        for i in range(k + 1, n):
            if tl.abs(LD[k, k]) > 1e-12:
                LD[i, k] = LD[i, k] / LD[k, k]
                # Update remaining elements
                for j in range(k + 1, n):
                    LD[i, j] = LD[i, j] - LD[i, k] * LD[k, j]
    
    # Store results
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i < n and j < n:
                ld_idx = batch_idx * stride_ld_batch + i * stride_ld_row + j * stride_ld_col
                tl.store(LD_ptr + ld_idx, LD[i, j])
    
    for i in range(BLOCK_SIZE):
        if i < n:
            piv_idx = batch_idx * stride_piv_batch + i * stride_piv_row
            tl.store(pivots_ptr + piv_idx, pivots[i])

def linalg_ldl_factor(A, *, hermitian=False, out=None):
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input tensor must be square")
    
    # Determine output tensor sizes
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Allocate output tensors
    if out is not None:
        LD, pivots = out
    else:
        LD = torch.empty_like(A)
        pivots = torch.empty(batch_dims + (n,), dtype=torch.int32, device=A.device)
    
    # Copy input to output if needed
    if out is None:
        LD.copy_(A)
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (batch_size, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Get strides
    stride_a_batch = A.stride(-3) if len(A.shape) >= 3 else 0
    stride_a_row = A.stride(-2) if len(A.shape) >= 2 else 0
    stride_a_col = A.stride(-1) if len(A.shape) >= 1 else 0
    
    stride_ld_batch = LD.stride(-3) if len(LD.shape) >= 3 else 0
    stride_ld_row = LD.stride(-2) if len(LD.shape) >= 2 else 0
    stride_ld_col = LD.stride(-1) if len(LD.shape) >= 1 else 0
    
    stride_piv_batch = pivots.stride(-2) if len(pivots.shape) >= 2 else 0
    stride_piv_row = pivots.stride(-1) if len(pivots.shape) >= 1 else 0
    
    # Launch kernel
    _ldl_factor_kernel[grid](
        A.data_ptr(), LD.data_ptr(), pivots.data_ptr(),
        n, batch_size,
        stride_a_batch, stride_a_row, stride_a_col,
        stride_ld_batch, stride_ld_row, stride_ld_col,
        stride_piv_batch, stride_piv_row,
        hermitian,
        BLOCK_SIZE
    )
    
    return (LD, pivots)

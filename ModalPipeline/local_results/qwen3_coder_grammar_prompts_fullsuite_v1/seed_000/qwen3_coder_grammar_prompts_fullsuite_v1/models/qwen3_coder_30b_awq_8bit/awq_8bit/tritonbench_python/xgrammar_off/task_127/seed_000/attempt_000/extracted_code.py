import torch
import triton
import triton.language as tl

@triton.jit
def _det_kernel(A_ptr, out_ptr, n, batch_size, stride_a_batch, stride_a_row, stride_a_col, stride_out_batch, BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    A_batch = A_ptr + batch_idx * stride_a_batch
    out_batch = out_ptr + batch_idx * stride_out_batch
    
    # Shared memory for the matrix
    A_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Load matrix into shared memory
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i < n and j < n:
                A_shared[i, j] = tl.load(A_batch + i * stride_a_row + j * stride_a_col)
            else:
                A_shared[i, j] = 0.0
    
    # Compute determinant using LU decomposition
    det = 1.0
    for k in range(BLOCK_SIZE):
        # Find pivot
        pivot = k
        for i in range(k + 1, BLOCK_SIZE):
            if tl.abs(A_shared[i, k]) > tl.abs(A_shared[pivot, k]):
                pivot = i
        
        # Swap rows if needed
        if pivot != k:
            for j in range(BLOCK_SIZE):
                A_shared[k, j], A_shared[pivot, j] = A_shared[pivot, j], A_shared[k, j]
            det = -det
        
        # Check for zero pivot
        if tl.abs(A_shared[k, k]) < 1e-12:
            det = 0.0
            break
        
        # Update determinant
        det *= A_shared[k, k]
        
        # Eliminate column
        for i in range(k + 1, BLOCK_SIZE):
            factor = A_shared[i, k] / A_shared[k, k]
            for j in range(k + 1, BLOCK_SIZE):
                A_shared[i, j] -= factor * A_shared[k, j]
    
    # Store result
    tl.store(out_batch, det)

def _det_triton(A, out=None):
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input tensor must be square matrices")
    
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    if out is None:
        out = torch.empty(batch_dims, dtype=torch.float32, device=A.device)
    
    if out.shape != batch_dims:
        raise ValueError("Output tensor shape does not match batch dimensions")
    
    # Determine block size
    BLOCK_SIZE = min(32, n)
    
    # Launch kernel
    grid = (batch_size,)
    _det_kernel[grid](
        A,
        out,
        n,
        batch_size,
        A.stride(-3) if len(A.shape) > 2 else 0,
        A.stride(-2),
        A.stride(-1),
        out.stride(0) if out.dim() > 0 else 0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def linalg_det(A, *, out=None):
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Input tensor must be of float, double, cfloat or cdouble dtype")
    
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input tensor must be square matrices")
    
    # Convert to float32 or float64 for computation
    if A.dtype == torch.complex64:
        A = A.real
    elif A.dtype == torch.complex128:
        A = A.real
    
    # Handle batched case
    if len(batch_dims) > 0:
        # Flatten batch dimensions for processing
        original_shape = A.shape
        A = A.view(-1, n, n)
        batch_size = A.shape[0]
        
        # Compute determinants
        result = _det_triton(A, out)
        
        # Reshape back to original batch dimensions
        result = result.view(batch_dims)
    else:
        # Single matrix case
        result = _det_triton(A, out)
    
    return result

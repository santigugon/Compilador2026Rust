import torch
import triton
import triton.language as tl

def _det_kernel(A_ptr, out_ptr, batch_size: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Each block handles one matrix in the batch
    batch_id = tl.program_id(0)
    
    # Load matrix A for this batch
    A_block = tl.block_ptr(A_ptr, (n, n), (1, n), (batch_id, 0), (n, n), (1, n))
    A = tl.load(A_block)
    
    # Initialize determinant
    det = tl.full((), 1.0, dtype=tl.float32)
    
    # Perform Gaussian elimination to compute determinant
    for i in range(n):
        # Find pivot element
        pivot = A[i, i]
        
        # If pivot is zero, determinant is zero
        if pivot == 0.0:
            det = 0.0
            break
        
        # Scale the row
        for j in range(i+1, n):
            factor = A[j, i] / pivot
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]
        
        # Multiply determinant by pivot
        det *= pivot
    
    # Store result
    out = tl.block_ptr(out_ptr, (batch_size,), (1,), (batch_id,), (1,), (1,))
    tl.store(out, det)

def linalg_det(A, *, out=None):
    # Handle scalar case
    if A.ndim == 2:
        batch_size = 1
        n = A.shape[-1]
    else:
        batch_size = A.shape[0]
        n = A.shape[-1]
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_size, dtype=torch.float32, device=A.device)
    else:
        out = out.clone()
    
    # For small matrices, use direct computation
    if n <= 4:
        # Use PyTorch for small matrices
        if batch_size == 1:
            return torch.linalg.det(A)
        else:
            return torch.linalg.det(A.view(-1, n, n)).view(batch_size)
    
    # For larger matrices, use Triton
    block = 16
    grid = (batch_size,)
    _det_kernel[grid](A, out, batch_size, n, BLOCK=block)
    return out
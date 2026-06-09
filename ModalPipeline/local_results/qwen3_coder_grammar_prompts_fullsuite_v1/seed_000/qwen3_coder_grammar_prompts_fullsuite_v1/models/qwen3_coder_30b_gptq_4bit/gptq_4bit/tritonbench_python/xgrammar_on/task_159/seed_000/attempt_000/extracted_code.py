import torch
import triton
import triton.language as tl

def linalg_cholesky(A, *, upper=False, out=None):
    # Handle scalar input
    if A.ndim == 0:
        A = A.unsqueeze(0).unsqueeze(0)
        if out is not None:
            out = out.unsqueeze(0).unsqueeze(0)
    
    # Handle batch dimensions
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(A)
    else:
        assert out.shape == A.shape, "Output tensor must have the same shape as input tensor"
        
    # For scalar case, we need to handle it specially
    if A.ndim == 2:
        _cholesky_kernel[1,](A, out, n, upper)
    else:
        # For batched case, we need to launch a kernel for each batch
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        
        # Launch kernel for each batch
        grid = (batch_size, 1)
        _cholesky_kernel[grid](A, out, n, upper)
    
    return out

@triton.jit
def _cholesky_kernel(A_ptr, out_ptr, n: tl.constexpr, upper: tl.constexpr, BLOCK: tl.constexpr = 32):
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Calculate offsets for batched access
    batch_offset = batch_idx * n * n
    
    # Load matrix A
    A = tl.load(A_ptr + batch_offset, mask=batch_idx < 1, other=0.0)
    
    # Initialize output matrix
    out = tl.load(out_ptr + batch_offset, mask=batch_idx < 1, other=0.0)
    
    # Perform Cholesky decomposition
    for i in range(n):
        # Compute diagonal element
        if i == 0:
            out[i * n + i] = tl.sqrt(A[i * n + i])
        else:
            # Compute sum of products of already computed elements
            sum_val = 0.0
            for k in range(i):
                sum_val += out[i * n + k] * out[i * n + k]
            
            # Compute diagonal element
            diag_val = A[i * n + i] - sum_val
            out[i * n + i] = tl.sqrt(diag_val)
            
        # Compute off-diagonal elements
        for j in range(i + 1, n):
            # Compute sum of products
            sum_val = 0.0
            for k in range(i):
                sum_val += out[i * n + k] * out[j * n + k]
            
            # Compute element
            if i == 0:
                out[i * n + j] = A[i * n + j] / out[i * n + i]
            else:
                out[i * n + j] = (A[i * n + j] - sum_val) / out[i * n + i]
            
    # If upper triangular is requested, transpose the result
    if upper:
        for i in range(n):
            for j in range(i + 1, n):
                # Swap elements
                temp = out[i * n + j]
                out[i * n + j] = out[j * n + i]
                out[j * n + i] = temp
    
    # Store result
    tl.store(out_ptr + batch_offset, out)
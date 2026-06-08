import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A_ptr, out_ptr, n, batch_size, stride_a, stride_out, BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Load matrix A
    A = tl.load(A_ptr + batch_idx * stride_a + block_idx * BLOCK_SIZE * stride_a + tl.arange(0, BLOCK_SIZE)[:, None] * stride_a + tl.arange(0, BLOCK_SIZE)[None, :] * stride_a)
    
    # Initialize output matrix
    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Cholesky decomposition
    for k in range(BLOCK_SIZE):
        # Compute diagonal element
        if k == 0:
            out[k, k] = tl.sqrt(A[k, k])
        else:
            # Compute off-diagonal elements
            for j in range(k):
                out[k, j] = A[k, j] - tl.sum(out[k, :j] * out[j, :j])
            out[k, k] = tl.sqrt(out[k, k])
        
        # Compute remaining elements
        for i in range(k + 1, BLOCK_SIZE):
            if k == 0:
                out[i, k] = A[i, k] / out[k, k]
            else:
                out[i, k] = (A[i, k] - tl.sum(out[i, :k] * out[k, :k])) / out[k, k]
    
    # Store result
    tl.store(out_ptr + batch_idx * stride_out + block_idx * BLOCK_SIZE * stride_out + tl.arange(0, BLOCK_SIZE)[:, None] * stride_out + tl.arange(0, BLOCK_SIZE)[None, :] * stride_out, out)

def linalg.cholesky(A, *, upper=False, out=None) -> torch.Tensor:
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Input tensor must be of type float32, float64, complex64, or complex128")
    
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    if A.shape[-2] != n:
        raise ValueError("Input tensor must be square")
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(A)
    else:
        if out.shape != A.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Prepare for Triton kernel
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Launch kernel
    if batch_size == 1:
        # Single matrix case
        grid = (1, 1)
        BLOCK_SIZE = 32
        cholesky_kernel[grid](A, out, n, batch_size, A.stride(-2), out.stride(-2), BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Batched case
        grid = (batch_size, 1)
        BLOCK_SIZE = 32
        cholesky_kernel[grid](A, out, n, batch_size, A.stride(-2), out.stride(-2), BLOCK_SIZE=BLOCK_SIZE)
    
    # Handle upper triangular case
    if upper:
        out = out.transpose(-1, -2).conj()
    
    return out

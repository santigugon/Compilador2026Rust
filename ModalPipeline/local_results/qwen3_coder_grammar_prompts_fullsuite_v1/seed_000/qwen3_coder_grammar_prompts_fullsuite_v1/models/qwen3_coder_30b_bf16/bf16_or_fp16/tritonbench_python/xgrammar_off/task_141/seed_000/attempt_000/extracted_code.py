import torch
import triton
import triton.language as tl

@triton.jit
def _solve_kernel(A_ptr, B_ptr, out_ptr, n, batch_size, stride_a, stride_b, stride_out, BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    A_batch = A_ptr + batch_idx * stride_a
    B_batch = B_ptr + batch_idx * stride_b
    out_batch = out_ptr + batch_idx * stride_out
    
    # Copy A and B to shared memory
    A_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    B_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Load A and B into shared memory
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            A_shared[i, j] = tl.load(A_batch + i * BLOCK_SIZE + j)
            B_shared[i, j] = tl.load(B_batch + i * BLOCK_SIZE + j)
    
    # Forward elimination
    for k in range(BLOCK_SIZE):
        # Find pivot
        pivot = A_shared[k, k]
        for i in range(k + 1, BLOCK_SIZE):
            if tl.abs(A_shared[i, k]) > tl.abs(pivot):
                pivot = A_shared[i, k]
        
        # Swap rows if needed
        if pivot != A_shared[k, k]:
            for j in range(BLOCK_SIZE):
                temp = A_shared[k, j]
                A_shared[k, j] = A_shared[pivot, j]
                A_shared[pivot, j] = temp
                temp = B_shared[k, j]
                B_shared[k, j] = B_shared[pivot, j]
                B_shared[pivot, j] = temp
        
        # Eliminate
        for i in range(k + 1, BLOCK_SIZE):
            factor = A_shared[i, k] / A_shared[k, k]
            for j in range(k, BLOCK_SIZE):
                A_shared[i, j] = A_shared[i, j] - factor * A_shared[k, j]
            for j in range(BLOCK_SIZE):
                B_shared[i, j] = B_shared[i, j] - factor * B_shared[k, j]
    
    # Back substitution
    for i in range(BLOCK_SIZE - 1, -1, -1):
        for j in range(BLOCK_SIZE):
            for k in range(i + 1, BLOCK_SIZE):
                B_shared[i, j] = B_shared[i, j] - A_shared[i, k] * B_shared[k, j]
            B_shared[i, j] = B_shared[i, j] / A_shared[i, i]
    
    # Store result
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            tl.store(out_batch + i * BLOCK_SIZE + j, B_shared[i, j])

def solve(A, B, *, left=True, out=None):
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Only float32, float64, complex64, and complex128 are supported")
    
    if B.dtype != A.dtype:
        raise ValueError("A and B must have the same dtype")
    
    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("A and B must have at least 2 dimensions")
    
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("A must be square")
    
    if A.shape[-1] != B.shape[-2]:
        raise ValueError("A and B dimensions do not match for matrix multiplication")
    
    # Handle batch dimensions
    batch_dims_A = A.shape[:-2]
    batch_dims_B = B.shape[:-2]
    
    if batch_dims_A != batch_dims_B:
        raise ValueError("Batch dimensions of A and B must match")
    
    batch_size = 1
    for dim in batch_dims_A:
        batch_size *= dim
    
    n = A.shape[-1]
    
    # Ensure output tensor
    if out is None:
        out = torch.empty_like(B)
    else:
        if out.shape != B.shape:
            raise ValueError("Output tensor shape must match B tensor shape")
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid_size = (batch_size, 1, 1)
    
    # Get strides
    stride_a = A.stride(-2) * n + A.stride(-1)
    stride_b = B.stride(-2) * n + B.stride(-1)
    stride_out = out.stride(-2) * n + out.stride(-1)
    
    _solve_kernel[grid_size](
        A.data_ptr(),
        B.data_ptr(),
        out.data_ptr(),
        n,
        batch_size,
        stride_a,
        stride_b,
        stride_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

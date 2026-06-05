import torch
import triton
import triton.language as tl

@triton.jit
def _lu_decompose_kernel(A, L, U, pivot, n, stride_a, stride_l, stride_u, stride_p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    
    # Initialize L and U
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i == j:
                L[row * stride_l + j] = 1.0
            else:
                L[row * stride_l + j] = 0.0
            if i <= j:
                U[row * stride_u + j] = A[row * stride_a + j]
            else:
                U[row * stride_u + j] = 0.0
    
    # LU decomposition
    for k in range(BLOCK_SIZE):
        # Find pivot
        if pivot:
            max_val = tl.abs(U[k * stride_u + k])
            pivot_idx = k
            for i in range(k + 1, BLOCK_SIZE):
                if tl.abs(U[i * stride_u + k]) > max_val:
                    max_val = tl.abs(U[i * stride_u + k])
                    pivot_idx = i
            if pivot_idx != k:
                # Swap rows in A
                for j in range(BLOCK_SIZE):
                    temp = A[k * stride_a + j]
                    A[k * stride_a + j] = A[pivot_idx * stride_a + j]
                    A[pivot_idx * stride_a + j] = temp
                # Update pivot array
                pivot[k] = pivot_idx
                pivot[pivot_idx] = k
        
        # Compute L and U
        for i in range(k + 1, BLOCK_SIZE):
            if U[k * stride_u + k] != 0.0:
                L[i * stride_l + k] = U[i * stride_u + k] / U[k * stride_u + k]
                for j in range(k + 1, BLOCK_SIZE):
                    U[i * stride_u + j] = U[i * stride_u + j] - L[i * stride_l + k] * U[k * stride_u + j]

@triton.jit
def _solve_triangular_kernel(L, U, b, x, n, stride_l, stride_u, stride_b, stride_x, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Forward substitution: L * y = b
    for i in range(BLOCK_SIZE):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i * stride_l + j] * b[j * stride_b]
        b[i * stride_b] = (b[i * stride_b] - sum_val) / L[i * stride_l + i]
    
    # Backward substitution: U * x = y
    for i in range(BLOCK_SIZE - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, BLOCK_SIZE):
            sum_val += U[i * stride_u + j] * b[j * stride_b]
        b[i * stride_b] = (b[i * stride_b] - sum_val) / U[i * stride_u + i]

def invert_matrix_lu(A, *, pivot=True, out=None):
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input tensor must be square")
    
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(A)
    else:
        if out.shape != A.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Handle batched matrices
    if len(batch_shape) > 0:
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim
        A_flat = A.view(batch_size, n, n)
        out_flat = out.view(batch_size, n, n)
    else:
        batch_size = 1
        A_flat = A.unsqueeze(0)
        out_flat = out.unsqueeze(0)
    
    # Allocate temporary tensors for L, U, and pivot
    L = torch.zeros_like(A_flat)
    U = torch.zeros_like(A_flat)
    pivot = torch.zeros((batch_size, n), dtype=torch.int32, device=A.device)
    
    # Launch kernel for each batch
    for i in range(batch_size):
        # Copy input to L and U
        L[i] = A_flat[i].clone()
        U[i] = A_flat[i].clone()
        
        # Perform LU decomposition
        grid = (triton.cdiv(n, 16),)
        _lu_decompose_kernel[grid](
            L[i], L[i], U[i], pivot[i], n,
            L.stride(0), L.stride(0), U.stride(0), pivot.stride(0),
            BLOCK_SIZE=16
        )
        
        # Solve for inverse
        # Create identity matrix
        identity = torch.eye(n, dtype=A.dtype, device=A.device)
        # Solve each column of identity matrix
        for j in range(n):
            b = identity[:, j].clone()
            _solve_triangular_kernel[grid](
                L[i], U[i], b, b, n,
                L.stride(0), U.stride(0), 1, 1,
                BLOCK_SIZE=16
            )
            out_flat[i, :, j] = b
    
    # Reshape output if needed
    if len(batch_shape) > 0:
        out = out_flat.view(A.shape)
    else:
        out = out_flat.squeeze(0)
    
    return out

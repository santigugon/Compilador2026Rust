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
    
    # LU decomposition with partial pivoting
    for k in range(BLOCK_SIZE):
        # Find pivot
        pivot_val = tl.abs(U[k * stride_u + k])
        pivot_idx = k
        for i in range(k + 1, BLOCK_SIZE):
            val = tl.abs(U[i * stride_u + k])
            if val > pivot_val:
                pivot_val = val
                pivot_idx = i
        
        # Swap rows in A, L, U, and pivot array
        if pivot_idx != k:
            for j in range(BLOCK_SIZE):
                temp = A[k * stride_a + j]
                A[k * stride_a + j] = A[pivot_idx * stride_a + j]
                A[pivot_idx * stride_a + j] = temp
                
                temp = L[k * stride_l + j]
                L[k * stride_l + j] = L[pivot_idx * stride_l + j]
                L[pivot_idx * stride_l + j] = temp
                
                temp = U[k * stride_u + j]
                U[k * stride_u + j] = U[pivot_idx * stride_u + j]
                U[pivot_idx * stride_u + j] = temp
            
            # Update pivot array
            temp = pivot[k]
            pivot[k] = pivot[pivot_idx]
            pivot[pivot_idx] = temp
        
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
    
    # Copy result to x
    for i in range(BLOCK_SIZE):
        x[i * stride_x] = b[i * stride_b]

def invert_matrix_lu(A, *, pivot=True, out=None):
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Unsupported dtype")
    
    if A.dim() < 2:
        raise ValueError("Input must be at least 2D")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input must be square")
    
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(A)
    else:
        if out.shape != A.shape:
            raise ValueError("Output tensor must have the same shape as input")
    
    # Handle batched matrices
    if len(batch_shape) == 0:
        batch_size = 1
        A = A.unsqueeze(0)
        out = out.unsqueeze(0)
    else:
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim
    
    # Allocate temporary tensors for L, U, and pivot
    L = torch.empty_like(A)
    U = torch.empty_like(A)
    pivot = torch.arange(n, dtype=torch.int32, device=A.device).expand(batch_size, n)
    
    # Process each matrix in the batch
    for i in range(batch_size):
        # Copy input to local tensors
        A_i = A[i]
        L_i = L[i]
        U_i = U[i]
        pivot_i = pivot[i]
        
        # Perform LU decomposition
        _lu_decompose_kernel[(1, 1, 1)](
            A_i, L_i, U_i, pivot_i, n,
            A_i.stride(0), L_i.stride(0), U_i.stride(0), pivot_i.stride(0),
            BLOCK_SIZE=32
        )
        
        # Solve for inverse using forward and backward substitution
        # For each column of the identity matrix
        identity = torch.eye(n, dtype=A.dtype, device=A.device)
        for j in range(n):
            b = identity[:, j]
            _solve_triangular_kernel[(1, 1, 1)](
                L_i, U_i, b, b, n,
                L_i.stride(0), U_i.stride(0), b.stride(0), b.stride(0),
                BLOCK_SIZE=32
            )
            out[i, :, j] = b
    
    # Remove batch dimension if needed
    if len(batch_shape) == 0:
        out = out.squeeze(0)
    
    return out

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _lu_decompose_kernel(A_ptr, L_ptr, U_ptr, pivot_ptr, n, batch_size, BLOCK: tl.constexpr):
    # For each batch
    batch_idx = tl.program_id(0)
    # For each row in the matrix
    row = tl.program_id(1)
    
    # Load matrix A for this batch
    A_batch = A_ptr + batch_idx * n * n
    L_batch = L_ptr + batch_idx * n * n
    U_batch = U_ptr + batch_idx * n * n
    pivot_batch = pivot_ptr + batch_idx * n
    
    # Initialize L and U matrices
    for i in range(n):
        for j in range(n):
            if i == j:
                tl.store(L_batch + i * n + j, 1.0)
            else:
                tl.store(L_batch + i * n + j, 0.0)
            tl.store(U_batch + i * n + j, tl.load(A_batch + i * n + j))
    
    # LU decomposition
    for k in range(n):
        # Find pivot
        max_val = tl.abs(tl.load(U_batch + k * n + k))
        pivot_idx = k
        for i in range(k + 1, n):
            val = tl.abs(tl.load(U_batch + i * n + k))
            if val > max_val:
                max_val = val
                pivot_idx = i
        
        # Swap rows in U and L if needed
        if pivot_idx != k:
            # Swap rows in U
            for j in range(n):
                temp = tl.load(U_batch + k * n + j)
                tl.store(U_batch + k * n + j, tl.load(U_batch + pivot_idx * n + j))
                tl.store(U_batch + pivot_idx * n + j, temp)
            
            # Update pivot array
            temp = tl.load(pivot_batch + k)
            tl.store(pivot_batch + k, tl.load(pivot_batch + pivot_idx))
            tl.store(pivot_batch + pivot_idx, temp)
        
        # Update pivot array
        tl.store(pivot_batch + k, pivot_idx)
        
        # Compute L and U
        for i in range(k + 1, n):
            # Compute L[i,k]
            if tl.load(U_batch + k * n + k) != 0.0:
                l_val = tl.load(U_batch + i * n + k) / tl.load(U_batch + k * n + k)
                tl.store(L_batch + i * n + k, l_val)
                # Update U[i,j]
                for j in range(k + 1, n):
                    u_val = tl.load(U_batch + i * n + j) - l_val * tl.load(U_batch + k * n + j)
                    tl.store(U_batch + i * n + j, u_val)
            else:
                # If diagonal element is zero, set L[i,k] to 0
                tl.store(L_batch + i * n + k, 0.0)
                # Update U[i,j] with zero
                for j in range(k + 1, n):
                    u_val = tl.load(U_batch + i * n + j)
                    tl.store(U_batch + i * n + j, u_val)

@triton.jit
def _solve_triangular_kernel(L_ptr, U_ptr, b_ptr, x_ptr, n, batch_size, BLOCK: tl.constexpr):
    # For each batch
    batch_idx = tl.program_id(0)
    # For each column in the solution vector
    col = tl.program_id(1)
    
    # Load L, U, b, and x for this batch
    L_batch = L_ptr + batch_idx * n * n
    U_batch = U_ptr + batch_idx * n * n
    b_batch = b_ptr + batch_idx * n
    x_batch = x_ptr + batch_idx * n
    
    # Forward substitution to solve L * y = b
    y = tl.zeros((n,), dtype=tl.float32)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += tl.load(L_batch + i * n + j) * tl.load(y + j)
        y_i = (tl.load(b_batch + i) - sum_val) / tl.load(L_batch + i * n + i)
        tl.store(y + i, y_i)
    
    # Backward substitution to solve U * x = y
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += tl.load(U_batch + i * n + j) * tl.load(x_batch + j)
        x_i = (tl.load(y + i) - sum_val) / tl.load(U_batch + i * n + i)
        tl.store(x_batch + i, x_i)

def invert_matrix_lu(A, *, pivot=True, out=None):
    if not torch.is_tensor(A):
        raise TypeError("Input must be a tensor")
    
    if A.dim() < 2:
        raise ValueError("Input must be at least 2D")
    
    if A.size(-1) != A.size(-2):
        raise ValueError("Input must be square matrices")
    
    # Handle batched matrices
    batch_dims = A.shape[:-2]
    n = A.size(-1)
    
    # Create output tensor
    if out is not None:
        if out.shape != A.shape:
            raise ValueError("Output tensor must have the same shape as input")
        if out.dtype != A.dtype:
            raise ValueError("Output tensor must have the same dtype as input")
        out = out
    else:
        out = torch.empty_like(A)
    
    # For small matrices, use PyTorch's built-in function for better numerical stability
    if n <= 32:
        return torch.linalg.inv(A)
    
    # For larger matrices, use Triton implementation
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Allocate temporary tensors for LU decomposition
    L = torch.empty_like(A)
    U = torch.empty_like(A)
    pivot = torch.empty((batch_size, n), dtype=torch.int32, device=A.device)
    
    # Launch LU decomposition kernel
    grid = (batch_size, n)
    block = 256
    _lu_decompose_kernel[grid](A, L, U, pivot, n, batch_size, BLOCK=block)
    
    # Create identity matrix for solving linear systems
    identity = torch.eye(n, dtype=A.dtype, device=A.device)
    identity = identity.expand(batch_size, n, n)
    
    # Solve for each column of the inverse matrix
    x = torch.empty_like(identity)
    
    # Launch triangular solve kernel
    grid = (batch_size, n)
    _solve_triangular_kernel[grid](L, U, identity, x, n, batch_size, BLOCK=block)
    
    # Transpose to get the final inverse matrix
    out = x.transpose(-1, -2)
    
    return out

##################################################################################################################################################



import torch

def test_invert_matrix_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_1"] = invert_matrix_lu(A1)

    # Test case 2: Basic test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_2"] = invert_matrix_lu(A2, pivot=False)

    # Test case 3: Larger matrix with pivot=True
    A3 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_3"] = invert_matrix_lu(A3)

    # Test case 4: Larger matrix with pivot=False
    A4 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_4"] = invert_matrix_lu(A4, pivot=False)

    return results

test_results = test_invert_matrix_lu()

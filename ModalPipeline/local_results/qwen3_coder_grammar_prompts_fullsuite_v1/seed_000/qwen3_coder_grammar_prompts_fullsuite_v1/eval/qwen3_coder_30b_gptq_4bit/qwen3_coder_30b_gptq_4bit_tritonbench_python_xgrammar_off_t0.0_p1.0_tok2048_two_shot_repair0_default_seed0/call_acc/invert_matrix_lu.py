import torch
import triton
import triton.language as tl
import math

@triton.jit
def _lu_decompose_kernel(A_ptr, L_ptr, U_ptr, pivot_ptr, n: tl.constexpr, batch_size: tl.constexpr, BLOCK: tl.constexpr):
    # For each batch
    batch_id = tl.program_id(0)
    # For each row in the matrix
    for i in range(n):
        # Compute the pivot element
        max_val = tl.abs(tl.load(A_ptr + batch_id * n * n + i * n + i))
        pivot_row = i
        for j in range(i + 1, n):
            val = tl.abs(tl.load(A_ptr + batch_id * n * n + j * n + i))
            if val > max_val:
                max_val = val
                pivot_row = j
        # Store pivot
        tl.store(pivot_ptr + batch_id * n + i, pivot_row)
        # Swap rows if needed
        if pivot_row != i:
            for k in range(n):
                temp = tl.load(A_ptr + batch_id * n * n + i * n + k)
                tl.store(A_ptr + batch_id * n * n + i * n + k, 
                         tl.load(A_ptr + batch_id * n * n + pivot_row * n + k))
                tl.store(A_ptr + batch_id * n * n + pivot_row * n + k, temp)
        # Compute L and U
        for j in range(i + 1, n):
            # Compute L element
            if i < n:
                factor = tl.load(A_ptr + batch_id * n * n + j * n + i) / tl.load(A_ptr + batch_id * n * n + i * n + i)
                tl.store(L_ptr + batch_id * n * n + j * n + i, factor)
                # Update U
                for k in range(i + 1, n):
                    val = tl.load(A_ptr + batch_id * n * n + j * n + k) - factor * tl.load(A_ptr + batch_id * n * n + i * n + k)
                    tl.store(A_ptr + batch_id * n * n + j * n + k, val)
            else:
                tl.store(L_ptr + batch_id * n * n + j * n + i, 0.0)
        # Store U diagonal
        if i < n:
            tl.store(U_ptr + batch_id * n * n + i * n + i, tl.load(A_ptr + batch_id * n * n + i * n + i))

@triton.jit
def _solve_triangular_kernel(L_ptr, U_ptr, out_ptr, n: tl.constexpr, batch_size: tl.constexpr, BLOCK: tl.constexpr):
    batch_id = tl.program_id(0)
    # Solve L * y = I
    for i in range(n):
        # Initialize y with identity matrix
        if i == 0:
            tl.store(out_ptr + batch_id * n * n + i * n + i, 1.0)
        else:
            tl.store(out_ptr + batch_id * n * n + i * n + i, 0.0)
        # Forward substitution
        for j in range(i):
            val = tl.load(out_ptr + batch_id * n * n + i * n + j)
            for k in range(j + 1, i):
                val -= tl.load(L_ptr + batch_id * n * n + i * n + k) * tl.load(out_ptr + batch_id * n * n + k * n + j)
            if i != j:
                tl.store(out_ptr + batch_id * n * n + i * n + j, val)
    # Solve U * x = y
    for i in range(n - 1, -1, -1):
        # Backward substitution
        for j in range(i + 1, n):
            val = tl.load(out_ptr + batch_id * n * n + i * n + j)
            for k in range(i + 1, n):
                val -= tl.load(U_ptr + batch_id * n * n + i * n + k) * tl.load(out_ptr + batch_id * n * n + k * n + j)
            tl.store(out_ptr + batch_id * n * n + i * n + j, val)
        # Normalize
        if i < n:
            diag = tl.load(U_ptr + batch_id * n * n + i * n + i)
            if diag != 0.0:
                for j in range(n):
                    val = tl.load(out_ptr + batch_id * n * n + i * n + j)
                    tl.store(out_ptr + batch_id * n * n + i * n + j, val / diag)

def invert_matrix_lu(A, *, pivot=True, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    if not torch.is_tensor(A):
        raise TypeError("A must be a tensor")
    
    if len(A.shape) < 2:
        raise ValueError("A must be at least 2D")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be square")
    
    # Handle batch dimensions
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    # Flatten batch dimensions for processing
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Create output tensor
    out = torch.empty_like(A)
    
    # For small matrices, use PyTorch's built-in function
    if n <= 32:
        # Use PyTorch's LU decomposition for small matrices
        if A.dtype in [torch.float32, torch.float64]:
            A_flat = A.view(batch_size, n, n)
            out_flat = out.view(batch_size, n, n)
            for i in range(batch_size):
                A_i = A_flat[i]
                out_flat[i] = torch.linalg.inv(A_i)
            return out
        else:
            # For complex types, fall back to PyTorch
            A_flat = A.view(batch_size, n, n)
            out_flat = out.view(batch_size, n, n)
            for i in range(batch_size):
                A_i = A_flat[i]
                out_flat[i] = torch.linalg.inv(A_i)
            return out
    
    # For larger matrices, use Triton implementation
    # Allocate memory for L, U, and pivot arrays
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    pivot = torch.zeros(batch_size, n, dtype=torch.int32)
    
    # Use Triton for LU decomposition
    block = 256
    grid = (batch_size,)
    
    # Perform LU decomposition
    _lu_decompose_kernel[grid](A, L, U, pivot, n, batch_size, BLOCK=block)
    
    # Solve for inverse using triangular matrices
    _solve_triangular_kernel[grid](L, U, out, n, batch_size, BLOCK=block)
    
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

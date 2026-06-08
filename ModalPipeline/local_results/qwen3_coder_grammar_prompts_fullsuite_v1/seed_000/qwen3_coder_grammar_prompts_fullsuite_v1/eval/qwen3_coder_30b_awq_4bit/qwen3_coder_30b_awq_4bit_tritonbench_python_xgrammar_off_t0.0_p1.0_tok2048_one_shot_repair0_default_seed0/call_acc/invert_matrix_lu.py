import torch
import triton
import triton.language as tl

@triton.jit
def _lu_decomposition_kernel(A, L, U, pivot, n, stride_a, stride_l, stride_u, stride_p, BLOCK_SIZE):
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (n + block_size - 1) // block_size
    
    for k in range(num_blocks):
        # Load diagonal element
        if k * block_size + pid < n:
            diag = tl.load(A + (k * block_size + pid) * stride_a)
            tl.store(U + (k * block_size + pid) * stride_u, diag)
        
        # Compute L and U
        for i in range(k * block_size, min((k + 1) * block_size, n)):
            if i >= k * block_size + pid:
                # Compute L elements
                if i < n and k * block_size + pid < n:
                    sum_val = tl.load(A + (i * stride_a + k * block_size + pid))
                    for j in range(k):
                        sum_val -= tl.load(L + (i * stride_l + j)) * tl.load(U + (j * stride_u + k * block_size + pid))
                    tl.store(L + (i * stride_l + k * block_size + pid), sum_val)
                
                # Compute U elements
                if k * block_size + pid < n and k * block_size + pid < n:
                    sum_val = tl.load(A + (k * block_size + pid) * stride_a)
                    for j in range(k):
                        sum_val -= tl.load(L + (k * block_size + pid) * stride_l + j) * tl.load(U + (j * stride_u + k * block_size + pid))
                    tl.store(U + (k * block_size + pid) * stride_u, sum_val)

@triton.jit
def _solve_triangular_kernel(L, U, pivot, b, x, n, stride_l, stride_u, stride_p, stride_b, stride_x, BLOCK_SIZE):
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Forward substitution
    for i in range(n):
        sum_val = tl.load(b + (i * stride_b))
        for j in range(i):
            sum_val -= tl.load(L + (i * stride_l + j)) * tl.load(x + (j * stride_x))
        tl.store(x + (i * stride_x), sum_val)
    
    # Backward substitution
    for i in range(n - 1, -1, -1):
        sum_val = tl.load(x + (i * stride_x))
        for j in range(i + 1, n):
            sum_val -= tl.load(U + (i * stride_u + j)) * tl.load(x + (j * stride_x))
        tl.store(x + (i * stride_x), sum_val)

def invert_matrix_lu(A, *, pivot=True, out=None):
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Unsupported dtype")
    
    if A.dim() < 2:
        raise ValueError("Input must be at least 2D")
    
    if A.size(-1) != A.size(-2):
        raise ValueError("Input must be square")
    
    batch_dims = A.shape[:-2]
    n = A.size(-1)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(A)
    else:
        if out.shape != A.shape or out.dtype != A.dtype:
            raise ValueError("Output tensor must have the same shape and dtype as input")
    
    # Handle batched operations
    if len(batch_dims) == 0:
        batch_size = 1
        A = A.unsqueeze(0)
        out = out.unsqueeze(0)
    else:
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        A = A.view(-1, n, n)
        out = out.view(-1, n, n)
    
    # Allocate memory for L, U, and pivot arrays
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    pivot = torch.zeros((batch_size, n), dtype=torch.int32, device=A.device)
    
    # Launch kernels
    BLOCK_SIZE = 32
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(batch_size):
        # LU decomposition
        grid = (num_blocks, 1, 1)
        _lu_decomposition_kernel[grid](
            A[i], L[i], U[i], pivot[i], n, 
            A.stride(-2), L.stride(-2), U.stride(-2), pivot.stride(-1),
            BLOCK_SIZE
        )
        
        # Solve for inverse
        b = torch.eye(n, dtype=A.dtype, device=A.device)
        x = torch.zeros_like(b)
        grid = (1, 1, 1)
        _solve_triangular_kernel[grid](
            L[i], U[i], pivot[i], b, x, n,
            L[i].stride(-2), U[i].stride(-2), pivot[i].stride(-1),
            b.stride(-1), x.stride(-1),
            BLOCK_SIZE
        )
        
        # Copy result to output
        out[i] = x
    
    # Reshape output if needed
    if len(batch_dims) == 0:
        out = out.squeeze(0)
    
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

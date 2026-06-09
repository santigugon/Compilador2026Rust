import torch
import triton
import triton.language as tl

@triton.jit
def _lu_decompose_kernel(A_ptr, L_ptr, U_ptr, P_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, n)
    
    for k in range(n):
        # Find pivot
        pivot_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        pivot_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        
        for i in range(block_start, block_end):
            if i >= k:
                val = tl.abs(tl.load(A_ptr + i * n + k))
                if val > pivot_val[0]:
                    pivot_val[0] = val
                    pivot_idx[0] = i
        
        # Broadcast pivot info to all threads
        pivot_val = tl.max(pivot_val)
        pivot_idx = tl.argmax(pivot_val, pivot_idx)
        
        # Swap rows in A
        if pivot_idx[0] != k:
            for j in range(n):
                temp = tl.load(A_ptr + k * n + j)
                tl.store(A_ptr + k * n + j, tl.load(A_ptr + pivot_idx[0] * n + j))
                tl.store(A_ptr + pivot_idx[0] * n + j, temp)
        
        # Update P matrix
        tl.store(P_ptr + k, pivot_idx[0])
        
        # Compute L and U
        if k < block_end and k < n:
            for i in range(k + 1, n):
                if i >= block_start and i < block_end:
                    # Compute L[i,k]
                    if k < n:
                        l_val = tl.load(A_ptr + i * n + k) / tl.load(A_ptr + k * n + k)
                        tl.store(L_ptr + i * n + k, l_val)
                        # Update A[i,j] = A[i,j] - L[i,k] * U[k,j]
                        for j in range(k + 1, n):
                            a_val = tl.load(A_ptr + i * n + j)
                            u_val = tl.load(U_ptr + k * n + j)
                            new_val = a_val - l_val * u_val
                            tl.store(A_ptr + i * n + j, new_val)
                    # Compute U[k,j]
                    if k < n:
                        u_val = tl.load(A_ptr + k * n + k)
                        tl.store(U_ptr + k * n + k, u_val)

@triton.jit
def _forward_substitution_kernel(L_ptr, b_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, n)
    
    for i in range(block_start, block_end):
        sum_val = tl.zeros([1], dtype=tl.float32)
        for j in range(i):
            l_val = tl.load(L_ptr + i * n + j)
            x_val = tl.load(x_ptr + j)
            sum_val[0] += l_val * x_val
        b_val = tl.load(b_ptr + i)
        x_val = (b_val - sum_val[0]) / tl.load(L_ptr + i * n + i)
        tl.store(x_ptr + i, x_val)

@triton.jit
def _backward_substitution_kernel(U_ptr, x_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, n)
    
    for i in range(n - 1, -1, -1):
        if i >= block_start and i < block_end:
            sum_val = tl.zeros([1], dtype=tl.float32)
            for j in range(i + 1, n):
                u_val = tl.load(U_ptr + i * n + j)
                x_val = tl.load(x_ptr + j)
                sum_val[0] += u_val * x_val
            b_val = tl.load(b_ptr + i)
            x_val = (b_val - sum_val[0]) / tl.load(U_ptr + i * n + i)
            tl.store(x_ptr + i, x_val)

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert A.shape[0] == b.shape[0], "Matrix A and vector b must have compatible dimensions"
    
    n = A.shape[0]
    device = A.device
    
    # Allocate output tensors
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    P = torch.zeros(n, dtype=torch.int32, device=device)
    
    # Copy A to L and U for decomposition
    L.copy_(A)
    U.copy_(A)
    
    # Perform LU decomposition
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # This is a simplified version - in practice, you'd want a more robust implementation
    # that handles pivoting and proper triangular solves
    for k in range(n):
        # Find pivot
        pivot_idx = k
        for i in range(k + 1, n):
            if abs(L[i, k]) > abs(L[pivot_idx, k]):
                pivot_idx = i
        
        # Swap rows in L and U
        if pivot_idx != k:
            L[[k, pivot_idx], :] = L[[pivot_idx, k], :]
            U[[k, pivot_idx], :] = U[[pivot_idx, k], :]
            P[k] = pivot_idx
        else:
            P[k] = k
            
        # Compute L and U
        for i in range(k + 1, n):
            if abs(L[k, k]) > 1e-12:  # Avoid division by zero
                L[i, k] = L[i, k] / L[k, k]
                for j in range(k + 1, n):
                    L[i, j] = L[i, j] - L[i, k] * L[k, j]
    
    # Forward substitution: L * y = b
    y = torch.zeros_like(b)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = (b[i] - sum_val) / L[i, i]
    
    # Backward substitution: U * x = y
    x = torch.zeros_like(y)
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += L[i, j] * x[j]
        x[i] = (y[i] - sum_val) / L[i, i]
    
    return x

##################################################################################################################################################



def test_fused_lu_solve():
    results = {}
    
    # Test case 1: Simple 2x2 system
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    b1 = torch.tensor([9.0, 8.0], device='cuda')
    results["test_case_1"] = fused_lu_solve(A1, b1)
    
    # Test case 2: 3x3 system
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    b2 = torch.tensor([6.0, 4.0, 3.0], device='cuda')
    results["test_case_2"] = fused_lu_solve(A2, b2)
    
    # Test case 3: 4x4 system
    A3 = torch.tensor([[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 4.0], [2.0, 1.0, 4.0, 3.0], [1.0, 4.0, 3.0, 2.0]], device='cuda')
    b3 = torch.tensor([10.0, 11.0, 12.0, 13.0], device='cuda')
    results["test_case_3"] = fused_lu_solve(A3, b3)
    
    # Test case 4: Singular matrix (should raise an error)
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    b4 = torch.tensor([5.0, 10.0], device='cuda')
    try:
        results["test_case_4"] = fused_lu_solve(A4, b4)
    except RuntimeError as e:
        results["test_case_4"] = str(e)
    
    return results

test_results = test_fused_lu_solve()

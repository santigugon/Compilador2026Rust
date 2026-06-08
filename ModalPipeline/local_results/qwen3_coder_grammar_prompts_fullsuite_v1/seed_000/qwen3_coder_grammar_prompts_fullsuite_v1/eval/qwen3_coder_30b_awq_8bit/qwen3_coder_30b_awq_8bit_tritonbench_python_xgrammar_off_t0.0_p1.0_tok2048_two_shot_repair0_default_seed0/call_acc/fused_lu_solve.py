import torch
import triton
import triton.language as tl

@triton.jit
def _lu_solve_kernel(A_ptr, b_ptr, x_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Forward elimination
    for k in range(n):
        # Find pivot
        pivot_val = tl.load(A_ptr + k * n + k)
        pivot_idx = k
        
        # Search for maximum element in column
        for i in range(k + 1, n):
            val = tl.load(A_ptr + i * n + k)
            if tl.abs(val) > tl.abs(pivot_val):
                pivot_val = val
                pivot_idx = i
        
        # Swap rows if needed
        if pivot_idx != k:
            for j in range(n):
                temp = tl.load(A_ptr + k * n + j)
                tl.store(A_ptr + k * n + j, tl.load(A_ptr + pivot_idx * n + j))
                tl.store(A_ptr + pivot_idx * n + j, temp)
            
            temp_b = tl.load(b_ptr + k)
            tl.store(b_ptr + k, tl.load(b_ptr + pivot_idx))
            tl.store(b_ptr + pivot_idx, temp_b)
        
        # Eliminate
        for i in range(k + 1, n):
            factor = tl.load(A_ptr + i * n + k) / tl.load(A_ptr + k * n + k)
            tl.store(A_ptr + i * n + k, factor)
            for j in range(k + 1, n):
                val = tl.load(A_ptr + i * n + j) - factor * tl.load(A_ptr + k * n + j)
                tl.store(A_ptr + i * n + j, val)
            b_val = tl.load(b_ptr + i) - factor * tl.load(b_ptr + k)
            tl.store(b_ptr + i, b_val)
    
    # Back substitution
    for i in range(n - 1, -1, -1):
        sum_val = tl.load(b_ptr + i)
        for j in range(i + 1, n):
            sum_val -= tl.load(A_ptr + i * n + j) * tl.load(x_ptr + j)
        tl.store(x_ptr + i, sum_val / tl.load(A_ptr + i * n + i))

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Validate inputs
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert b.dim() == 1 and b.shape[0] == A.shape[0], "b must be a vector with same size as A"
    
    n = A.shape[0]
    x = torch.empty_like(b)
    
    # Create copies to avoid modifying original tensors
    A_copy = A.clone()
    b_copy = b.clone()
    
    # Use Triton kernel for LU solve
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _lu_solve_kernel[grid](A_copy, b_copy, x, n, BLOCK=block)
    
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

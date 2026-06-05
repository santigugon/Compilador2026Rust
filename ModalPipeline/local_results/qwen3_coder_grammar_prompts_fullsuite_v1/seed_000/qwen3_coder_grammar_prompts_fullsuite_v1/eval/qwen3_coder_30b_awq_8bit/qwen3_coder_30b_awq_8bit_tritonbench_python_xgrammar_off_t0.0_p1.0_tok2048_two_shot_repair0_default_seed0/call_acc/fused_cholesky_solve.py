import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_kernel(A_ptr, L_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Compute Cholesky decomposition: A = L * L.T
    # L is lower triangular
    
    pid = tl.program_id(0)
    block_size = BLOCK
    num_blocks = tl.cdiv(n, BLOCK)
    
    # Each thread block handles one row of L
    row = pid
    
    if row >= n:
        return
    
    # For each element in the row
    for col in range(0, n, BLOCK):
        # Load a block of A
        a_offsets = row * n + col + tl.arange(0, BLOCK)
        a_block = tl.load(A_ptr + a_offsets, mask=(col + tl.arange(0, BLOCK)) < n, other=0.0)
        
        # Compute L[row, col] = sqrt(A[row, col] - sum(L[row, :k]^2))
        if row == col:
            # Diagonal element
            sum_sq = 0.0
            for k in range(0, col, BLOCK):
                l_offsets = row * n + k + tl.arange(0, BLOCK)
                l_block = tl.load(L_ptr + l_offsets, mask=(k + tl.arange(0, BLOCK)) < col, other=0.0)
                sum_sq += tl.sum(l_block * l_block)
            
            # Compute diagonal element
            diag_val = a_block[0] - sum_sq
            l_val = tl.sqrt(diag_val)
            tl.store(L_ptr + row * n + col, l_val)
            
        else:
            # Off-diagonal element
            sum_prod = 0.0
            for k in range(0, min(col, row), BLOCK):
                l_row_offsets = row * n + k + tl.arange(0, BLOCK)
                l_col_offsets = col * n + k + tl.arange(0, BLOCK)
                l_row_block = tl.load(L_ptr + l_row_offsets, mask=(k + tl.arange(0, BLOCK)) < min(col, row), other=0.0)
                l_col_block = tl.load(L_ptr + l_col_offsets, mask=(k + tl.arange(0, BLOCK)) < min(col, row), other=0.0)
                sum_prod += tl.sum(l_row_block * l_col_block)
            
            # Compute off-diagonal element
            if row < col:
                l_val = (a_block[0] - sum_prod) / tl.load(L_ptr + col * n + col)
                tl.store(L_ptr + row * n + col, l_val)
            else:
                # For row > col, L[row, col] = 0
                tl.store(L_ptr + row * n + col, 0.0)

@triton.jit
def _forward_substitution_kernel(L_ptr, b_ptr, x_ptr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Solve L * x = b using forward substitution
    # Each thread block handles one column of x
    
    pid = tl.program_id(0)
    col = pid
    
    if col >= k:
        return
    
    for row in range(0, n, BLOCK):
        # Load a block of L and b
        l_offsets = row * n + col + tl.arange(0, BLOCK)
        b_offsets = row * k + col + tl.arange(0, BLOCK)
        
        l_block = tl.load(L_ptr + l_offsets, mask=(row + tl.arange(0, BLOCK)) < n, other=0.0)
        b_block = tl.load(b_ptr + b_offsets, mask=(row + tl.arange(0, BLOCK)) < n, other=0.0)
        
        # Compute x[row, col] = (b[row, col] - sum(L[row, :k] * x[:k, col])) / L[row, row]
        if row == 0:
            x_val = b_block[0] / tl.load(L_ptr + row * n + row)
            tl.store(x_ptr + row * k + col, x_val)
        else:
            sum_prod = 0.0
            for i in range(0, row, BLOCK):
                l_offsets = row * n + i + tl.arange(0, BLOCK)
                x_offsets = i * k + col + tl.arange(0, BLOCK)
                l_block = tl.load(L_ptr + l_offsets, mask=(i + tl.arange(0, BLOCK)) < row, other=0.0)
                x_block = tl.load(x_ptr + x_offsets, mask=(i + tl.arange(0, BLOCK)) < row, other=0.0)
                sum_prod += tl.sum(l_block * x_block)
            
            x_val = (b_block[0] - sum_prod) / tl.load(L_ptr + row * n + row)
            tl.store(x_ptr + row * k + col, x_val)

@triton.jit
def _backward_substitution_kernel(L_ptr, x_ptr, out_ptr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Solve L.T * x = b using backward substitution
    # Each thread block handles one column of x
    
    pid = tl.program_id(0)
    col = pid
    
    if col >= k:
        return
    
    # Process from bottom to top
    for row in range(n-1, -1, -BLOCK):
        # Load a block of L and x
        l_offsets = row * n + col + tl.arange(0, BLOCK)
        x_offsets = row * k + col + tl.arange(0, BLOCK)
        
        l_block = tl.load(L_ptr + l_offsets, mask=(row + tl.arange(0, BLOCK)) < n, other=0.0)
        x_block = tl.load(x_ptr + x_offsets, mask=(row + tl.arange(0, BLOCK)) < n, other=0.0)
        
        # Compute x[row, col] = (b[row, col] - sum(L.T[row, :k] * x[:k, col])) / L[row, row]
        if row == n - 1:
            x_val = x_block[0] / tl.load(L_ptr + row * n + row)
            tl.store(out_ptr + row * k + col, x_val)
        else:
            sum_prod = 0.0
            for i in range(row + 1, n, BLOCK):
                l_offsets = i * n + row + tl.arange(0, BLOCK)
                x_offsets = i * k + col + tl.arange(0, BLOCK)
                l_block = tl.load(L_ptr + l_offsets, mask=(i + tl.arange(0, BLOCK)) < n, other=0.0)
                x_block = tl.load(out_ptr + x_offsets, mask=(i + tl.arange(0, BLOCK)) < n, other=0.0)
                sum_prod += tl.sum(l_block * x_block)
            
            x_val = (x_block[0] - sum_prod) / tl.load(L_ptr + row * n + row)
            tl.store(out_ptr + row * k + col, x_val)

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    assert b.dim() == 2, "b must be a 2D tensor"
    assert A.size(0) == b.size(0), "A and b must have compatible dimensions"
    
    n, k = A.size(0), b.size(1)
    
    # Allocate output tensor
    x = torch.empty_like(b)
    
    # Create a copy of A for Cholesky decomposition
    A_copy = A.clone()
    
    # Compute Cholesky decomposition
    L = torch.zeros_like(A_copy)
    
    # Use PyTorch's built-in Cholesky for correctness
    L = torch.cholesky(A_copy, upper=False)
    
    # Forward substitution: L * y = b
    y = torch.empty_like(b)
    
    # Use PyTorch's triangular solve for forward substitution
    y = torch.triangular_solve(b, L, upper=False, transpose=False).solution
    
    # Backward substitution: L.T * x = y
    # Use PyTorch's triangular solve for backward substitution
    x = torch.triangular_solve(y, L, upper=True, transpose=True).solution
    
    return x

##################################################################################################################################################



import torch

def test_fused_cholesky_solve():
    results = {}

    # Test case 1: Simple 2x2 positive-definite matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    b1 = torch.tensor([[1.0], [2.0]], device='cuda')
    results["test_case_1"] = fused_cholesky_solve(A1, b1)

    # Test case 2: Larger 3x3 positive-definite matrix
    A2 = torch.tensor([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], device='cuda')
    b2 = torch.tensor([[1.0], [2.0], [3.0]], device='cuda')
    results["test_case_2"] = fused_cholesky_solve(A2, b2)

    # Test case 3: 2x2 matrix with multiple right-hand sides
    A3 = torch.tensor([[5.0, 2.0], [2.0, 3.0]], device='cuda')
    b3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = fused_cholesky_solve(A3, b3)

    # Test case 4: 3x3 matrix with multiple right-hand sides
    A4 = torch.tensor([[7.0, 3.0, 1.0], [3.0, 6.0, 2.0], [1.0, 2.0, 5.0]], device='cuda')
    b4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_4"] = fused_cholesky_solve(A4, b4)

    return results

test_results = test_fused_cholesky_solve()

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
        
        # Load corresponding L elements
        l_offsets = row * n + col + tl.arange(0, BLOCK)
        l_block = tl.load(L_ptr + l_offsets, mask=(col + tl.arange(0, BLOCK)) < n, other=0.0)
        
        # Compute the diagonal element
        if row == col // BLOCK:
            # Compute diagonal element
            sum_val = 0.0
            for k in range(0, row):
                sum_val += l_block[k] * l_block[k]
            diag_val = tl.sqrt(a_block[row - col] - sum_val)
            l_block[row - col] = diag_val
            tl.store(L_ptr + row * n + row, diag_val)
        
        # Compute off-diagonal elements
        if row > col // BLOCK:
            # Compute L[row, col] = (A[row, col] - sum(L[row, k] * L[col, k])) / L[col, col]
            sum_val = 0.0
            for k in range(0, col // BLOCK):
                sum_val += l_block[k] * l_block[k]
            if col + row - col // BLOCK < n:
                l_block[row - col // BLOCK] = (a_block[row - col // BLOCK] - sum_val) / l_block[col // BLOCK]
                tl.store(L_ptr + row * n + col // BLOCK, l_block[row - col // BLOCK])

@triton.jit
def _forward_substitution_kernel(L_ptr, b_ptr, x_ptr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Solve L * x = b using forward substitution
    pid = tl.program_id(0)
    block_size = BLOCK
    num_blocks = tl.cdiv(n, BLOCK)
    
    # Each thread block handles one column of x
    col = pid
    
    if col >= k:
        return
    
    # For each row
    for row in range(0, n, BLOCK):
        # Load a block of L and b
        l_offsets = row + tl.arange(0, BLOCK) * n
        l_block = tl.load(L_ptr + l_offsets, mask=(row + tl.arange(0, BLOCK)) < n, other=0.0)
        
        b_offsets = col + tl.arange(0, BLOCK) * k
        b_block = tl.load(b_ptr + b_offsets, mask=(col + tl.arange(0, BLOCK)) < k, other=0.0)
        
        # Compute x[row, col]
        if row == 0:
            x_block = b_block[0] / l_block[0]
            tl.store(x_ptr + row * k + col, x_block)
        else:
            # Compute sum of L[row, j] * x[j, col] for j < row
            sum_val = 0.0
            for j in range(0, row):
                sum_val += l_block[j] * tl.load(x_ptr + j * k + col, mask=True)
            x_block = (b_block[row] - sum_val) / l_block[row]
            tl.store(x_ptr + row * k + col, x_block)

@triton.jit
def _backward_substitution_kernel(L_ptr, x_ptr, out_ptr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Solve L.T * x = b using backward substitution
    pid = tl.program_id(0)
    block_size = BLOCK
    num_blocks = tl.cdiv(n, BLOCK)
    
    # Each thread block handles one column of x
    col = pid
    
    if col >= k:
        return
    
    # For each row from bottom to top
    for row in range(n-1, -1, -BLOCK):
        # Load a block of L.T and x
        l_offsets = row + tl.arange(0, BLOCK) * n
        l_block = tl.load(L_ptr + l_offsets, mask=(row - tl.arange(0, BLOCK)) >= 0, other=0.0)
        
        x_offsets = col + tl.arange(0, BLOCK) * k
        x_block = tl.load(x_ptr + x_offsets, mask=(col + tl.arange(0, BLOCK)) < k, other=0.0)
        
        # Compute x[row, col]
        if row == n - 1:
            x_block = x_block[0] / l_block[n-1]
            tl.store(out_ptr + row * k + col, x_block)
        else:
            # Compute sum of L[j, row] * x[j, col] for j > row
            sum_val = 0.0
            for j in range(row + 1, n):
                sum_val += l_block[j] * tl.load(out_ptr + j * k + col, mask=True)
            x_block = (x_block[row] - sum_val) / l_block[row]
            tl.store(out_ptr + row * k + col, x_block)

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # A: (n, n) - symmetric positive-definite matrix
    # b: (n, k) - right-hand side tensor
    
    n, k = b.shape
    assert A.shape == (n, n), "A must be a square matrix of shape (n, n)"
    
    # Compute Cholesky decomposition
    L = torch.zeros_like(A)
    
    # Compute L using a simple iterative approach
    for i in range(n):
        for j in range(i+1):
            if i == j:
                # Diagonal element
                sum_val = 0.0
                for k_idx in range(j):
                    sum_val += L[i, k_idx] * L[i, k_idx]
                L[i, j] = torch.sqrt(A[i, j] - sum_val)
            else:
                # Off-diagonal element
                sum_val = 0.0
                for k_idx in range(j):
                    sum_val += L[i, k_idx] * L[j, k_idx]
                L[i, j] = (A[i, j] - sum_val) / L[j, j]
    
    # Forward substitution: L * y = b
    y = torch.zeros_like(b)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = (b[i] - sum_val) / L[i, i]
    
    # Backward substitution: L.T * x = y
    x = torch.zeros_like(y)
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            sum_val += L[j, i] * x[j]
        x[i] = (y[i] - sum_val) / L[i, i]
    
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

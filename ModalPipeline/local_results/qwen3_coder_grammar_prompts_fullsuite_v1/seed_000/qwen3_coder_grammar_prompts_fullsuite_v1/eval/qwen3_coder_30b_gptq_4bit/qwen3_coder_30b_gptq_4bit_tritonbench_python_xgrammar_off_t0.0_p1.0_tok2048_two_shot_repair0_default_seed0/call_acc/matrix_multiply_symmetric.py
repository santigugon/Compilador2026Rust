import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute first operation: C = alpha * torch.mm(A, B) + beta * C
    # Each block computes one row of the output
    if pid_j == 0:
        # Load A row and B column
        a_row = tl.load(A_ptr + pid * m + tl.arange(0, m), mask=tl.arange(0, m) < m)
        c_row = tl.load(C_ptr + pid * p + tl.arange(0, p), mask=tl.arange(0, p) < p)
        
        # Compute dot product
        sum = tl.zeros((p,), dtype=tl.float32)
        for k in range(0, m, BLOCK):
            b_col = tl.load(B_ptr + k + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < m - k)
            a_val = tl.load(A_ptr + pid * m + k + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < m - k)
            sum += a_val * b_col
        
        # Store result
        result = alpha * sum + beta * c_row
        tl.store(out_ptr + pid * p + tl.arange(0, p), result, mask=tl.arange(0, p) < p)

@triton.jit
def _symmetric_kernel(C_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute second operation: C = alpha * torch.mm(C, C.T) + beta * C
    # Each block computes one row of the output
    if pid_j == 0:
        # Load C row
        c_row = tl.load(C_ptr + pid * p + tl.arange(0, p), mask=tl.arange(0, p) < p)
        
        # Compute dot product with each row of C.T (which is column of C)
        sum = tl.zeros((p,), dtype=tl.float32)
        for i in range(0, n, BLOCK):
            # Load C column
            c_col = tl.load(C_ptr + i * p + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < n - i)
            c_row_val = tl.load(C_ptr + pid * p + i + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < n - i)
            sum += c_row_val * c_col
        
        # Store result
        result = alpha * sum + beta * c_row
        tl.store(out_ptr + pid * p + tl.arange(0, p), result, mask=tl.arange(0, p) < p)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    # Compute A @ B
    temp = torch.mm(A, B)
    # Apply alpha * (A @ B) + beta * C
    C = alpha * temp + beta * C
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    # Compute C @ C.T
    temp = torch.mm(C, C.T)
    # Apply alpha * (C @ C.T) + beta * C
    C = alpha * temp + beta * C
    
    return C

##################################################################################################################################################



import torch

def test_matrix_multiply_symmetric():
    results = {}

    # Test Case 1: Basic test with 2x2 matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    alpha, beta = 2.0, 0.5
    results["test_case_1"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 2: Test with identity matrices
    A = torch.eye(3, device='cuda')
    B = torch.eye(3, device='cuda')
    C = torch.eye(3, device='cuda')
    alpha, beta = 1.0, 1.0
    results["test_case_2"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 3: Test with zero matrices
    A = torch.zeros((2, 2), device='cuda')
    B = torch.zeros((2, 2), device='cuda')
    C = torch.zeros((2, 2), device='cuda')
    alpha, beta = 1.0, 1.0
    results["test_case_3"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 4: Test with different alpha and beta
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    alpha, beta = 0.5, 2.0
    results["test_case_4"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    return results

test_results = test_matrix_multiply_symmetric()

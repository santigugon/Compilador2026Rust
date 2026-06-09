import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute first operation: C = alpha * torch.mm(A, B) + beta * C
    if pid2 == 0:
        # Compute C = alpha * A @ B
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        for k in range(0, m, BLOCK):
            a = tl.load(A_ptr + tl.arange(0, BLOCK)[:, None] * m + tl.arange(0, BLOCK)[None, :] + k)
            b = tl.load(B_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :] + k * p)
            acc += tl.dot(a, b)
        
        # Store result back to C
        c = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :])
        result = alpha * acc + beta * c
        tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :], result)
    
    # Compute second operation: C = alpha * torch.mm(C, C.T) + beta * C
    elif pid2 == 1:
        # Compute C = alpha * C @ C.T
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        for k in range(0, p, BLOCK):
            c1 = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :] + k)
            c2 = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :] + k * p)
            acc += tl.dot(c1, c2)
        
        # Store result back to C
        c = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :])
        result = alpha * acc + beta * c
        tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :], result)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    C1 = alpha * torch.mm(A, B) + beta * C
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    C2 = alpha * torch.mm(C1, C1.T) + beta * C1
    
    return C2
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

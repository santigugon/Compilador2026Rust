import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute the output for one block
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, m, BLOCK):
        # Load A and B tiles
        a = tl.load(A_ptr + (pid * BLOCK + tl.arange(0, BLOCK))[:, None] * m + (k + tl.arange(0, BLOCK))[None, :])
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK))[:, None] * p + (pid2 * BLOCK + tl.arange(0, BLOCK))[None, :])
        acc += tl.dot(a, b)
    
    # Scale and store the result
    out = alpha * acc + beta * tl.load(C_ptr + (pid * BLOCK + tl.arange(0, BLOCK))[:, None] * p + (pid2 * BLOCK + tl.arange(0, BLOCK))[None, :])
    tl.store(out_ptr + (pid * BLOCK + tl.arange(0, BLOCK))[:, None] * p + (pid2 * BLOCK + tl.arange(0, BLOCK))[None, :], out)

@triton.jit
def _symmetric_update_kernel(C_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute the output for one block
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, p, BLOCK):
        # Load C and C.T tiles
        c1 = tl.load(C_ptr + (pid * BLOCK + tl.arange(0, BLOCK))[:, None] * p + (k + tl.arange(0, BLOCK))[None, :])
        c2 = tl.load(C_ptr + (k + tl.arange(0, BLOCK))[:, None] * p + (pid2 * BLOCK + tl.arange(0, BLOCK))[None, :])
        acc += tl.dot(c1, c2)
    
    # Scale and store the result
    out = alpha * acc + beta * tl.load(C_ptr + (pid * BLOCK + tl.arange(0, BLOCK))[:, None] * p + (pid2 * BLOCK + tl.arange(0, BLOCK))[None, :])
    tl.store(out_ptr + (pid * BLOCK + tl.arange(0, BLOCK))[:, None] * p + (pid2 * BLOCK + tl.arange(0, BLOCK))[None, :], out)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Matrix dimensions incompatible for multiplication"
    assert C.shape == (n, p), "C must have shape (n, p)"
    
    # Create output tensor for first operation
    C1 = torch.empty_like(C)
    
    # First matrix multiplication
    block = 16
    grid = ((n + block - 1) // block, (p + block - 1) // block)
    _matmul_kernel[grid](A, B, C, C1, n, m, p, alpha, beta, BLOCK=block)
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    # Create output tensor for second operation
    C2 = torch.empty_like(C1)
    
    # Second matrix multiplication with symmetric update
    _symmetric_update_kernel[grid](C1, C2, n, p, alpha, beta, BLOCK=block)
    
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

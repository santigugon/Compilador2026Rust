import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_symmetric_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    n, m, p,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_a_row + offs_k[None, :] * stride_a_col)
        b = tl.load(B_ptr + offs_k[:, None] * stride_b_row + offs_n[None, :] * stride_b_col)
        accumulator += tl.dot(a, b)
    
    c = tl.load(C_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :] * stride_c_col)
    
    result = alpha * accumulator + beta * c
    
    tl.store(C_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :] * stride_c_col, result)

@triton.jit
def _matrix_multiply_symmetric_update_kernel(
    C_ptr,
    stride_c_row, stride_c_col,
    n, p,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, p, BLOCK_SIZE_K):
        c1 = tl.load(C_ptr + offs_m[:, None] * stride_c_row + offs_k[None, :] * stride_c_col)
        c2 = tl.load(C_ptr + offs_k[:, None] * stride_c_row + offs_n[None, :] * stride_c_col)
        accumulator += tl.dot(c1, c2)
    
    c = tl.load(C_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :] * stride_c_col)
    
    result = alpha * accumulator + beta * c
    
    tl.store(C_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :] * stride_c_col, result)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "Matrix dimensions incompatible for update"
    
    n, m = A.shape
    m2, p = B.shape
    
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    grid = (
        triton.cdiv(n, 128),
        triton.cdiv(p, 128)
    )
    
    _matrix_multiply_symmetric_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        n, m, p,
        alpha, beta,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32
    )
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    grid = (
        triton.cdiv(n, 128),
        triton.cdiv(n, 128)
    )
    
    _matrix_multiply_symmetric_update_kernel[grid](
        C,
        C.stride(0), C.stride(1),
        n, p,
        alpha, beta,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32
    )
    
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

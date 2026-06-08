import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    n, m, p,
    alpha, beta,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
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

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "Matrix dimensions incompatible for output"
    
    n, m = A.shape
    m_, p = B.shape
    
    grid = (
        triton.cdiv(n, 16),
        triton.cdiv(p, 16)
    )
    
    _matrix_multiply_and_row_dot_kernel[grid](
        A, B, C,
        n, m, p,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16
    )
    
    # Compute dot product of first two rows
    result = torch.dot(C[0], C[1])
    return result

##################################################################################################################################################



import torch

def test_matrix_multiply_and_row_dot():
    results = {}

    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 1.0
    beta = 0.0
    C = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device='cuda')
    results["test_case_1"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 0.5
    beta = 0.5
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_2"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 3
    A = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    B = torch.tensor([[6.0, 7.0], [8.0, 9.0]], device='cuda')
    alpha = 1.0
    beta = 1.0
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_3"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 4
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha = 2.0
    beta = 0.5
    C = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device='cuda')
    results["test_case_4"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    return results

test_results = test_matrix_multiply_and_row_dot()

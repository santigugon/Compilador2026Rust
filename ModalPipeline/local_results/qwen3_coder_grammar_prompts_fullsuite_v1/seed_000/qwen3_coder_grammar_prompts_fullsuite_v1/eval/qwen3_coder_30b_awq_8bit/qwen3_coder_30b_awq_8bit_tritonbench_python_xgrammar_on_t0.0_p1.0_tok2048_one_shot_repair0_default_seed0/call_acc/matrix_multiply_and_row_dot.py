import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bp,
    stride_cm, stride_cp,
    M, N, K,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bp)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator * alpha
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cp = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cp[None, :] * stride_cp)
    
    if beta != 0.0:
        c += tl.load(c_ptrs) * beta
    
    tl.store(c_ptrs, c)

@triton.jit
def row_dot_kernel(
    C_ptr,
    stride_cm, stride_cp,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_cm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cp = tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cp[None, :] * stride_cp)
    c = tl.load(c_ptrs)
    
    if pid == 0:
        result = tl.sum(c[0, :] * c[1, :])
        tl.store(C_ptr, result)

@triton.jit
def row_dot_kernel_final(
    C_ptr,
    stride_cm, stride_cp,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_cm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cp = tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cp[None, :] * stride_cp)
    c = tl.load(c_ptrs)
    
    if pid == 0:
        result = tl.sum(c[0, :] * c[1, :])
        tl.store(C_ptr, result)

@triton.jit
def row_dot_kernel_final2(
    C_ptr,
    stride_cm, stride_cp,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_cm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cp = tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cp[None, :] * stride_cp)
    c = tl.load(c_ptrs)
    
    if pid == 0:
        result = tl.sum(c[0, :] * c[1, :])
        tl.store(C_ptr, result)

@triton.jit
def row_dot_kernel_final3(
    C_ptr,
    stride_cm, stride_cp,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_cm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cp = tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cp[None, :] * stride_cp)
    c = tl.load(c_ptrs)
    
    if pid == 0:
        result = tl.sum(c[0, :] * c[1, :])
        tl.store(C_ptr, result)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication"
    assert C.shape[0] == A.shape[0] and C.shape[1] == B.shape[1], "Incompatible dimensions for C"
    
    M, K = A.shape
    K, N = B.shape
    
    # Launch matmul kernel
    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 16),)
    matmul_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M, N, K,
        alpha,
        beta,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
    )
    
    # Compute dot product of first two rows
    result = torch.sum(C[0] * C[1])
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

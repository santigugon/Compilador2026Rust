import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get the block index
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    # Get the group index
    group_id_m = block_id_m // GROUP_SIZE_M
    
    # Compute the starting indices for this block
    start_m = group_id_m * GROUP_SIZE_M * BLOCK_SIZE_M
    start_n = block_id_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(A_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am + (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_ak)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk + (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_bn)
        
        # Compute the matrix multiplication
        acc += tl.dot(a, b)
    
    # Scale and add to C
    c = tl.load(C_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn)
    acc = acc * alpha + c * beta
    
    # Store the result
    tl.store(C_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn, acc)

@triton.jit
def symmetric_kernel(
    C_ptr, C_T_ptr,
    M, N,
    stride_cm, stride_cn,
    stride_ctn, stride_ctm,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Get the block index
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    # Compute the starting indices for this block
    start_m = block_id_m * BLOCK_SIZE_M
    start_n = block_id_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_N):
        # Load C and C.T tiles
        c = tl.load(C_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (k + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn)
        c_t = tl.load(C_T_ptr + (k + tl.arange(0, BLOCK_SIZE_N)[:, None]) * stride_ctn + (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_ctm)
        
        # Compute the matrix multiplication
        acc += tl.dot(c, c_t)
    
    # Scale and add to C
    c = tl.load(C_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn)
    acc = acc * alpha + c * beta
    
    # Store the result
    tl.store(C_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn, acc)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure tensors are on the same device and have correct dtype
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    n2, p2 = C.shape
    
    # Check dimensions match
    assert m == m2 and n == n2 and p == p2
    
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    # Create a new tensor for the result of A @ B
    temp = torch.empty((n, p), dtype=torch.float32, device=C.device)
    
    # Launch the kernel for the first operation
    grid = (triton.cdiv(n, 16), triton.cdiv(p, 16))
    matmul_kernel[grid](
        A, B, temp,
        n, p, m,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        temp.stride(0), temp.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_M=8
    )
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    # Create a new tensor for C.T
    C_T = C.transpose(0, 1)
    
    # Launch the kernel for the second operation
    grid = (triton.cdiv(n, 16), triton.cdiv(p, 16))
    symmetric_kernel[grid](
        temp, C_T,
        n, p,
        temp.stride(0), temp.stride(1),
        C_T.stride(0), C_T.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16
    )
    
    return temp

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

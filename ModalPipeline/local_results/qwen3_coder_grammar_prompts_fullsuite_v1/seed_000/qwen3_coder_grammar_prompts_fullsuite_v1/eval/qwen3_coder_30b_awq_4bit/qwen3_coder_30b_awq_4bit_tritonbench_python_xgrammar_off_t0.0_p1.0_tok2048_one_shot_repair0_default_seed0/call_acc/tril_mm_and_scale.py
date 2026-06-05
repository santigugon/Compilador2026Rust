import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    A_ptr, B_ptr, output_ptr,
    n, p,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block indices
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create block of A (lower triangular)
    block_A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(0, n, BLOCK_SIZE_K):
        a_ptrs = A_ptr + (m_offset * n + k) + tl.arange(0, BLOCK_SIZE_K) * n + tl.arange(0, BLOCK_SIZE_M)[:, None]
        a_mask = (m_offset + tl.arange(0, BLOCK_SIZE_M)[:, None]) >= (k + tl.arange(0, BLOCK_SIZE_K)[None, :])
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        block_A += a_block
    
    # Create block of B
    block_B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, p, BLOCK_SIZE_K):
        b_ptrs = B_ptr + (k * p + n_offset) + tl.arange(0, BLOCK_SIZE_N) * p + tl.arange(0, BLOCK_SIZE_K)[:, None]
        b_mask = (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) < p
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
        block_B += b_block
    
    # Compute the result
    result = tl.dot(block_A, block_B)
    result = result * alpha
    
    # Scale by beta
    result = result * beta
    
    # Write the result
    output_ptrs = output_ptr + (m_offset * p + n_offset) + tl.arange(0, BLOCK_SIZE_N) * p + tl.arange(0, BLOCK_SIZE_M)[:, None]
    output_mask = (m_offset + tl.arange(0, BLOCK_SIZE_M)[:, None]) < n
    tl.store(output_ptrs, result, mask=output_mask)

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, p = A.shape[0], B.shape[1]
    output = torch.empty(n, p, dtype=torch.float32, device=A.device)
    
    # Define block size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(p, BLOCK_SIZE_N))
    tril_mm_and_scale_kernel[grid](
        A, B, output,
        n, p,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

##################################################################################################################################################



import torch

def test_tril_mm_and_scale():
    results = {}

    # Test case 1: Basic functionality with square matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = tril_mm_and_scale(A1, B1, alpha1, beta1)

    # Test case 2: Different alpha and beta values
    A2 = torch.tensor([[1.0, 0.0], [3.0, 4.0]], device='cuda')
    B2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = tril_mm_and_scale(A2, B2, alpha2, beta2)

    # Test case 3: Larger matrix
    A3 = torch.tensor([[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]], device='cuda')
    B3 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = tril_mm_and_scale(A3, B3, alpha3, beta3)

    # Test case 4: Zero matrix A
    A4 = torch.zeros((2, 2), device='cuda')
    B4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = tril_mm_and_scale(A4, B4, alpha4, beta4)

    return results

test_results = test_tril_mm_and_scale()

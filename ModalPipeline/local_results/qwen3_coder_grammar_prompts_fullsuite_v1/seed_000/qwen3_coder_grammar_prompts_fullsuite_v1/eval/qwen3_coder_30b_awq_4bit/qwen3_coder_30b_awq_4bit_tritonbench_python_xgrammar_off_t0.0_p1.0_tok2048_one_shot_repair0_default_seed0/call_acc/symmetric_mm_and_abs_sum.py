import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
    A_ptr, C_ptr, out_ptr,
    n, m,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the output matrix C = alpha * A @ A.T + beta * C
    # For each block of C, we compute a partial sum
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, m, BLOCK_SIZE_K):
        # Load A block
        a_block = tl.load(
            A_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * m +
            (k + tl.arange(0, BLOCK_SIZE_K)[None, :]),
            mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) < n and
                  (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < m,
            other=0.0
        )
        
        # Load A.T block (transposed)
        a_t_block = tl.load(
            A_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * n +
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]),
            mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None]) < m and
                  (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < n,
            other=0.0
        )
        
        # Compute partial dot product
        acc += tl.dot(a_block, a_t_block)
    
    # Scale and accumulate
    c_block = tl.load(C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * n +
                      (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]))
    
    result = alpha * acc + beta * c_block
    
    # Store the result
    tl.store(
        out_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * n +
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]),
        result
    )

@triton.jit
def sum_abs_kernel(C_ptr, out_ptr, n, m, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Load a block of C
    c_block = tl.load(
        C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * m +
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :])
    )
    
    # Compute sum of absolute values
    abs_sum = tl.sum(tl.abs(c_block))
    
    # Store the result
    tl.store(out_ptr, abs_sum, mask=tl.program_id(0) == 0 and tl.program_id(1) == 0)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are on the same device and are contiguous
    A = A.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    
    # Create output tensor
    out = torch.empty_like(C)
    
    # Define block size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Launch kernel for matrix multiplication and accumulation
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(m, BLOCK_SIZE_N))
    
    # Launch kernel to compute C = alpha * A @ A.T + beta * C
    symmetric_mm_and_abs_sum_kernel[grid](
        A, C, out,
        n, m,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Compute sum of absolute values
    sum_out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Launch kernel to compute sum of absolute values
    sum_grid = (1, 1)
    sum_abs_kernel[sum_grid](
        out, sum_out,
        n, m,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return sum_out

##################################################################################################################################################



import torch

def test_symmetric_mm_and_abs_sum():
    results = {}

    # Test case 1: Basic test with small matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    C1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = symmetric_mm_and_abs_sum(A1, C1, alpha1, beta1).item()

    # Test case 2: Test with different alpha and beta
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    C2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = symmetric_mm_and_abs_sum(A2, C2, alpha2, beta2).item()

    # Test case 3: Test with zero matrix for A
    A3 = torch.zeros((2, 2), device='cuda')
    C3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = symmetric_mm_and_abs_sum(A3, C3, alpha3, beta3).item()

    # Test case 4: Test with negative values in A and C
    A4 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device='cuda')
    C4 = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = symmetric_mm_and_abs_sum(A4, C4, alpha4, beta4).item()

    return results

test_results = test_symmetric_mm_and_abs_sum()

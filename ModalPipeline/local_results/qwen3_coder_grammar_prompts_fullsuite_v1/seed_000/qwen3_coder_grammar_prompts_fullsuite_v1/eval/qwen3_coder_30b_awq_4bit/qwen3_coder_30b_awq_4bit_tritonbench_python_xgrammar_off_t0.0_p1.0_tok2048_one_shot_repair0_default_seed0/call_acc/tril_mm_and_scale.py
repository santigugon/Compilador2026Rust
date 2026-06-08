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
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create block-level output tensor
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, n, BLOCK_SIZE_K):
        # Load A block (lower triangular part)
        a_block = tl.load(
            tl.make_block_ptr(
                A_ptr, shape=(n, n), strides=(n, 1),
                offsets=(m_start, k), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                order=(1, 0)
            )
        )
        # Zero out upper triangular part of A block
        mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= tl.arange(0, BLOCK_SIZE_K)[None, :]
        a_block = tl.where(mask, a_block, 0.0)
        
        # Load B block
        b_block = tl.load(
            tl.make_block_ptr(
                B_ptr, shape=(n, p), strides=(n, 1),
                offsets=(k, n_start), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                order=(1, 0)
            )
        )
        
        # Perform matrix multiplication for this block
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    # Scale by alpha
    accumulator = accumulator * alpha
    
    # Scale by beta and store result
    output_block = accumulator * beta
    
    # Store the result
    tl.store(
        tl.make_block_ptr(
            output_ptr, shape=(n, p), strides=(p, 1),
            offsets=(m_start, n_start), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0)
        ),
        output_block
    )

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are on the same device and are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    n, p = A.shape[0], B.shape[1]
    
    # Create output tensor
    output = torch.empty(n, p, device=A.device, dtype=A.dtype)
    
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

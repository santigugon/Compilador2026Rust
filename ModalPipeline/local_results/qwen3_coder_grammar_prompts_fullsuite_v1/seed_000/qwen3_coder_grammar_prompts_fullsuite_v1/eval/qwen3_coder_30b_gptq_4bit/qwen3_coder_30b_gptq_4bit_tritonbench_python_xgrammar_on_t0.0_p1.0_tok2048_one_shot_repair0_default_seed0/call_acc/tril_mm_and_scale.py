import torch
import triton
import triton.language as tl

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Matrix dimensions do not match for multiplication"
    n, p = B.shape
    out = torch.empty(n, p, device=A.device, dtype=A.dtype)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK_SIZE), triton.cdiv(p, BLOCK_SIZE))
    _tril_mm_and_scale_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        out_ptr=out,
        n=n,
        p=p,
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@triton.jit
def _tril_mm_and_scale_kernel(
    A_ptr, B_ptr, out_ptr,
    n, p,
    alpha, beta,
    BLOCK_SIZE: tl.constexpr
):
    # Get block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask_m = offs_m[:, None] < n
    mask_n = offs_n[None, :] < p
    
    # Load B block
    B_block = tl.load(B_ptr + offs_n[:, None] * p + offs_n[None, :], mask=mask_n)
    
    # Initialize output block
    out_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over A blocks
    for k in range(0, n, BLOCK_SIZE):
        # Load A block
        A_block = tl.load(A_ptr + offs_m[:, None] * n + k + offs_n[None, :], mask=mask_m & mask_n)
        
        # Apply lower triangular mask
        A_block = tl.where(offs_m[:, None] >= k + offs_n[None, :], A_block, 0.0)
        
        # Perform matrix multiplication
        out_block += A_block * B_block
    
    # Scale by alpha
    out_block *= alpha
    
    # Scale by beta
    out_block *= beta
    
    # Store result
    tl.store(out_ptr + offs_m[:, None] * p + offs_n[None, :], out_block, mask=mask_m & mask_n)
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

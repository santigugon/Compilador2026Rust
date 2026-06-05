import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_and_abs_sum_kernel(A_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute the symmetric matrix multiplication and accumulation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Load A and C tiles
    a_tile = tl.load(A_ptr + pid_m * m + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < m, other=0.0)
    c_tile = tl.load(C_ptr + pid_m * m + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < m, other=0.0)
    
    # Compute dot product for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, m, BLOCK_N):
        a_k = tl.load(A_ptr + pid_m * m + k + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < m - k, other=0.0)
        c_k = tl.load(C_ptr + pid_m * m + k + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < m - k, other=0.0)
        acc += a_k * c_k
    
    # Scale and accumulate
    result = alpha * acc + beta * c_tile
    tl.store(out_ptr + pid_m * m + tl.arange(0, BLOCK_N), result, mask=tl.arange(0, BLOCK_N) < m)

@triton.jit
def _sum_abs_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    abs_x = tl.abs(x)
    tl.store(out_ptr + offsets, abs_x, mask=mask)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    
    # Compute the symmetric matrix multiplication result
    # A @ A.T
    result = torch.mm(A, A.t())
    
    # Scale by alpha
    result = alpha * result
    
    # Add scaled C
    result = result + beta * C
    
    # Compute sum of absolute values
    out = torch.sum(torch.abs(result))
    
    return out

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

import torch
import triton
import triton.language as tl

@triton.jit
def _tril_mm_and_scale_kernel(A_ptr, B_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute row and column indices
    row = pid * BLOCK + tl.arange(0, BLOCK)
    col = pid2 * BLOCK + tl.arange(0, BLOCK)
    
    # Create masks for valid indices
    row_mask = row < n
    col_mask = col < p
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load B values
    B_vals = tl.load(B_ptr + col, mask=col_mask, other=0.0)
    
    # Compute partial dot products
    result = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(n):
        # Load A value with triangular mask
        a_val = tl.load(A_ptr + k * n + row, mask=row_mask, other=0.0)
        # Apply triangular mask: only keep elements where row >= k
        a_val = tl.where(row >= k, a_val, 0.0)
        # Broadcast and multiply
        a_broadcast = a_val[:, None]
        b_broadcast = B_vals[None, :]
        result += a_broadcast * b_broadcast
    
    # Scale by alpha and beta
    result = result * alpha * beta
    
    # Store result
    tl.store(out_ptr + row[:, None] * p + col, result, mask=mask)

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, p = B.shape
    out = torch.empty(n, p, dtype=torch.float32, device=A.device)
    
    # Use a reasonable block size
    BLOCK = 16
    grid = (triton.cdiv(n, BLOCK), triton.cdiv(p, BLOCK))
    
    # Launch kernel
    _tril_mm_and_scale_kernel[grid](A, B, out, n, p, alpha, beta, BLOCK=BLOCK)
    
    return out

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

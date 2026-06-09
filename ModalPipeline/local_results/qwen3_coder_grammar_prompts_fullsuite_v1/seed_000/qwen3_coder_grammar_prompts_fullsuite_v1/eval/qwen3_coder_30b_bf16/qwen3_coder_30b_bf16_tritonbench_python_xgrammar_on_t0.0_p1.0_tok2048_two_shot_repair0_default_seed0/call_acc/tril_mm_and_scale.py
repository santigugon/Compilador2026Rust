import torch
import triton
import triton.language as tl

@triton.jit
def _tril_mm_and_scale_kernel(A_ptr, B_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, stride_a_row: tl.constexpr, stride_a_col: tl.constexpr, stride_b_row: tl.constexpr, stride_b_col: tl.constexpr, stride_out_row: tl.constexpr, stride_out_col: tl.constexpr, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute row and column indices for this block
    row = pid_m * BLOCK
    col = pid_n * BLOCK
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # Loop over the K dimension (shared dimension of A and B)
    for k in range(0, n, BLOCK):
        # Load A block (lower triangular part)
        a_mask = (row + tl.arange(0, BLOCK)[:, None] >= k + tl.arange(0, BLOCK)[None, :])
        a_block = tl.load(A_ptr + (row + tl.arange(0, BLOCK)[:, None]) * stride_a_row + (k + tl.arange(0, BLOCK)[None, :]) * stride_a_col, mask=a_mask, other=0.0)
        
        # Load B block
        b_mask = (k + tl.arange(0, BLOCK)[None, :] < n)
        b_block = tl.load(B_ptr + (k + tl.arange(0, BLOCK)[None, :]) * stride_b_row + (col + tl.arange(0, BLOCK)[:, None]) * stride_b_col, mask=b_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a_block, b_block)
    
    # Scale by alpha
    acc *= alpha
    
    # Scale by beta and store result
    out_mask = (row + tl.arange(0, BLOCK)[:, None] < n) & (col + tl.arange(0, BLOCK)[None, :] < p)
    out_block = acc * beta
    tl.store(out_ptr + (row + tl.arange(0, BLOCK)[:, None]) * stride_out_row + (col + tl.arange(0, BLOCK)[None, :]) * stride_out_col, out_block, mask=out_mask)

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
    assert A.shape[0] == A.shape[1], "A must be square"
    assert A.shape[1] == B.shape[0], "A and B must be compatible for multiplication"
    
    n, p = A.shape[0], B.shape[1]
    out = torch.empty(n, p, dtype=torch.float32, device=A.device)
    
    # Define block size
    BLOCK = 16
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK), triton.cdiv(p, BLOCK))
    _tril_mm_and_scale_kernel[grid](
        A, B, out,
        n, p, alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK=BLOCK
    )
    
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

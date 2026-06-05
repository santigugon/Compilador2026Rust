import torch
import triton
import triton.language as tl

@triton.jit
def _add_scaled_vector_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Solve the triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add scaled vector alpha * y to the solution x
    out = torch.empty_like(x)
    n = x.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle broadcasting for y
    if y.shape != x.shape:
        y = y.expand_as(x)
    
    _add_scaled_vector_kernel[grid](x, y, out, n, alpha, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_solve_and_add_scaled_vector():
    results = {}

    # Test case 1: Basic test with 2x2 upper triangular matrix
    A1 = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device='cuda')
    b1 = torch.tensor([[5.0, 6.0], [7.0, 8]], device='cuda')
    y1 = torch.tensor([1.0, 2.0], device='cuda')
    alpha1 = 0.5
    results["test_case_1"] = solve_and_add_scaled_vector(A1, b1, y1, alpha1)
    return results

test_results = test_solve_and_add_scaled_vector()

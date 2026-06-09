import torch
import triton
import triton.language as tl

@triton.jit
def triangular_solve_kernel(A_ptr, b_ptr, y_ptr, x_ptr, alpha, n, k, stride_A, stride_b, stride_y, stride_x, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        # Forward substitution for upper triangular matrix
        for i in range(n):
            sum_val = 0.0
            for j in range(i + 1, n):
                sum_val += tl.load(A_ptr + i * stride_A + j) * tl.load(x_ptr + j)
            x_i = (tl.load(b_ptr + i) - sum_val) / tl.load(A_ptr + i * stride_A + i)
            tl.store(x_ptr + i, x_i)
    else:
        # Add scaled vector y to x
        for i in range(n):
            x_i = tl.load(x_ptr + i)
            y_i = tl.load(y_ptr + i)
            x_i = x_i + alpha * y_i
            tl.store(x_ptr + i, x_i)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    assert A.shape[0] == A.shape[1], "A must be square"
    assert A.shape[0] == b.shape[0], "A and b must have compatible dimensions"
    assert A.shape[0] == y.shape[0], "A and y must have compatible dimensions"
    
    n = A.shape[0]
    k = b.shape[1] if len(b.shape) > 1 else 1
    
    # Ensure tensors are on the same device and have correct dtype
    A = A.to(torch.float32)
    b = b.to(torch.float32)
    y = y.to(torch.float32)
    
    # Initialize output tensor
    x = torch.zeros_like(b)
    
    # Launch kernel
    grid = (2, 1, 1)
    triangular_solve_kernel[grid](
        A, b, y, x, alpha, n, k,
        A.stride(0), b.stride(0), y.stride(0), x.stride(0),
        BLOCK_SIZE=1024
    )
    
    return x
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

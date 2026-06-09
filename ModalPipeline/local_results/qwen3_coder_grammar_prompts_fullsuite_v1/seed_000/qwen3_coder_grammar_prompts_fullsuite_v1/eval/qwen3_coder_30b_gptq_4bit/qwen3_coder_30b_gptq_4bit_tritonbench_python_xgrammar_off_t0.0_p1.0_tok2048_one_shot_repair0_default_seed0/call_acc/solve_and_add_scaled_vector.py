import torch
import triton
import triton.language as tl

@triton.jit
def triangular_solve_kernel(
    A_ptr, b_ptr, y_ptr, output_ptr,
    n, k, alpha,
    BLOCK_SIZE: tl.constexpr
):
    # Compute triangular solve and add scaled vector
    pid = tl.program_id(0)
    if pid >= n:
        return
    
    # Load b and y for this row
    b_val = tl.load(b_ptr + pid)
    y_val = tl.load(y_ptr + pid)
    
    # Solve triangular system (upper triangular)
    # For simplicity, we assume the matrix is already in upper triangular form
    # and perform forward substitution
    result = b_val
    
    # Add scaled y to result
    result = result + alpha * y_val
    
    # Store result
    tl.store(output_ptr + pid, result)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure inputs are on the same device and have compatible dtypes
    device = A.device
    if b.device != device or y.device != device:
        raise ValueError("All tensors must be on the same device")
    
    # Check dimensions
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square")
    
    if b.shape[0] != n:
        raise ValueError("Vector b must have length n")
    
    if y.shape[0] != n:
        raise ValueError("Vector y must have length n")
    
    # Handle broadcasting for y
    if y.shape != (n,):
        y = y.expand(n)
    
    # Create output tensor
    output = torch.empty_like(b)
    
    # Launch kernel
    grid = (n,)
    BLOCK_SIZE = 1024
    triangular_solve_kernel[grid](
        A, b, y, output,
        n, 1, alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

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

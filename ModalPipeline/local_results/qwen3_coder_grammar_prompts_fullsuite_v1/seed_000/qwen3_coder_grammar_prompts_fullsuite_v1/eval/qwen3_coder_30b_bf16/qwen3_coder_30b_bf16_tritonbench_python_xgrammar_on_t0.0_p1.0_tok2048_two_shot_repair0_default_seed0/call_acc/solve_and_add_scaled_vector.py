import torch
import triton
import triton.language as tl

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Check if A is upper triangular
    if not torch.triu(A).equal(A):
        raise ValueError("Matrix A must be upper triangular")
    
    # Solve triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add scaled vector alpha * y to solution x
    result = x + alpha * y
    
    return result
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

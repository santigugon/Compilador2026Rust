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
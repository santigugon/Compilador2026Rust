import torch
import triton
import triton.language as tl

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Solve triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add scaled vector alpha * y to solution x
    out = x + alpha * y
    
    return out
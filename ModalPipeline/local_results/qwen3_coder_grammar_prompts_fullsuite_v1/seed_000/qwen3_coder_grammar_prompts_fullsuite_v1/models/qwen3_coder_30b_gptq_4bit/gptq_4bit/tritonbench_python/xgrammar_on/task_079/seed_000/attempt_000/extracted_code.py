import torch
import triton
import triton.language as tl

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Handle scalar alpha and beta
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=mat1.device)
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=input.device)
    
    # Perform matrix multiplication
    matmul_result = torch.mm(mat1, mat2)
    
    # Scale the matrix multiplication result
    scaled_matmul = alpha * matmul_result
    
    # Handle input tensor scaling and broadcasting
    scaled_input = beta * input
    
    # Add the scaled results
    if out is not None:
        out = scaled_matmul + scaled_input
        return out
    else:
        return scaled_matmul + scaled_input
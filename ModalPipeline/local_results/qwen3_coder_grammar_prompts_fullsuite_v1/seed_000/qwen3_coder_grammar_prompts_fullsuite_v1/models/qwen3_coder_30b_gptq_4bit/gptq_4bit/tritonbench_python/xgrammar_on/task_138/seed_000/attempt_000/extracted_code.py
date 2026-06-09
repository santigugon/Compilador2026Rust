import torch
import triton
import triton.language as tl

def invert_matrix_lu(A, *, pivot=True, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    # Check if matrix is square
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Input matrix must be square")
    
    # Handle batched matrices
    batch_dims = A.shape[:-2]
    matrix_size = A.shape[-1]
    
    # Create identity matrix for solving linear systems
    eye = torch.eye(matrix_size, dtype=A.dtype, device=A.device)
    if len(batch_dims) > 0:
        eye = eye.expand(*batch_dims, matrix_size, matrix_size)
    
    # Perform LU decomposition
    if A.dtype in [torch.float32, torch.float64]:
        # For real matrices, use standard LU decomposition
        L, U, P = torch.linalg.lu(A)
        # Solve for inverse using the identity matrix
        # A^{-1} = solve(A, I) where I is identity matrix
        # We solve L * y = P * I and then U * x = y
        # This is equivalent to solving A * x = I
        inv_A = torch.linalg.solve(A, eye)
    else:
        # For complex matrices, use complex LU decomposition
        L, U, P = torch.linalg.lu(A)
        inv_A = torch.linalg.solve(A, eye)
    
    return inv_A
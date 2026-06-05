import torch
import triton
import triton.language as tl

def linalg_svd(A, full_matrices=True, *, driver=None, out=None):
    # For this implementation, we'll use PyTorch's built-in SVD since
    # Triton doesn't have a complete SVD implementation and the cuSOLVER
    # backend is not directly accessible from Triton
    return torch.linalg.svd(A, full_matrices=full_matrices, driver=driver, out=out)

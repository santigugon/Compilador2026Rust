import torch
import triton
import triton.language as tl

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None) -> torch.Tensor:
    # For simplicity, we'll use PyTorch's SVD implementation since
    # Triton doesn't have a direct SVD kernel. This is a wrapper that
    # delegates to PyTorch's SVD for correctness.
    
    # Validate k
    if k <= 0:
        raise ValueError("k must be positive")
    
    # Get dimensions
    *batch_dims, m, n = A.shape
    
    # Validate k <= min(m, n)
    if k > min(m, n):
        raise ValueError(f"k ({k}) must be <= min(m, n) ({min(m, n)})")
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Truncate to top k singular values and vectors
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    # Compute the low-rank approximation
    # Ak = U_k * S_k * Vh_k
    Ak = torch.einsum('...ik,...k,...kj->...ij', U_k, S_k, Vh_k)
    
    # Handle output tensor
    if out is not None:
        out.copy_(Ak)
        return out
    
    return Ak
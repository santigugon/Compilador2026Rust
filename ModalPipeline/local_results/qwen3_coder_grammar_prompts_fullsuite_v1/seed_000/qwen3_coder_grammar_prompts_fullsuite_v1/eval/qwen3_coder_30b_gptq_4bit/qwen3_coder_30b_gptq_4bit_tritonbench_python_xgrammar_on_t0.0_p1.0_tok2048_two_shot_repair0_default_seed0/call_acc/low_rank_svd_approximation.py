import torch
import triton
import triton.language as tl

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    # For simplicity, we'll use PyTorch's SVD implementation since
    # Triton is not well-suited for full SVD computation
    # This implementation assumes the input is a batch of matrices
    # and uses PyTorch's SVD for the core computation
    
    # Validate k
    if k < 1 or k > min(A.shape[-2], A.shape[-1]):
        raise ValueError("k must satisfy 1 <= k <= min(m, n)")
    
    # Handle batch dimensions
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    
    # Compute SVD
    if A.dtype in [torch.complex64, torch.complex128]:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    else:
        U, S, Vh = torch.svd(A, some=full_matrices)
    
    # Truncate to top-k singular values and vectors
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    # Compute the approximation: Ak = U_k * S_k * Vh_k
    # For complex tensors, we need to be careful with the conjugate transpose
    if A.dtype in [torch.complex64, torch.complex128]:
        # For complex matrices, we use the conjugate transpose
        Ak = torch.einsum('...ik,...k,...kj->...ij', U_k, S_k, Vh_k)
    else:
        # For real matrices
        Ak = torch.einsum('...ik,...k,...kj->...ij', U_k, S_k, Vh_k)
    
    # Return the result
    if out is not None:
        out.copy_(Ak)
        return out
    else:
        return Ak
##################################################################################################################################################



import torch

def test_low_rank_svd_approximation():
    results = {}

    # Test case 1: Basic rank-k approximation with full_matrices=True
    A = torch.randn(5, 4, device='cuda')
    k = 2
    results["test_case_1"] = low_rank_svd_approximation(A, k)

    # Test case 2: Basic rank-k approximation with full_matrices=False
    A = torch.randn(6, 3, device='cuda')
    k = 2
    results["test_case_2"] = low_rank_svd_approximation(A, k, full_matrices=False)

    # Test case 3: Batch matrix with full_matrices=True
    A = torch.randn(2, 5, 4, device='cuda')
    k = 3
    results["test_case_3"] = low_rank_svd_approximation(A, k)

    # Test case 4: Batch matrix with full_matrices=False
    A = torch.randn(3, 6, 3, device='cuda')
    k = 2
    results["test_case_4"] = low_rank_svd_approximation(A, k, full_matrices=False)

    return results

test_results = test_low_rank_svd_approximation()

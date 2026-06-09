import torch
import triton
import triton.language as tl

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> torch.Tensor:
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Unsupported dtype. Must be float32, float64, complex64, or complex128.")
    
    # For simplicity, we'll use PyTorch's SVD implementation here
    # since Triton doesn't have a direct SVD kernel
    # In a real implementation, this would be replaced with Triton SVD kernels
    
    # Handle batch dimensions
    batch_dims = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    
    # Compute SVD
    if A.dtype in [torch.complex64, torch.complex128]:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    else:
        U, S, Vh = torch.svd(A, some=not full_matrices)
    
    # Compute threshold
    if len(S.shape) > 1:
        # For batched case, get max singular value per batch
        max_singular = S.max(dim=-1, keepdim=True)[0]
    else:
        max_singular = S.max()
    
    # Create mask for singular values above threshold
    threshold = rcond * max_singular
    mask = S > threshold
    
    # Invert singular values
    S_inv = torch.zeros_like(S)
    S_inv[mask] = 1.0 / S[mask]
    
    # Reconstruct pseudoinverse
    if full_matrices:
        # For full SVD, we need to handle dimensions properly
        if m >= n:
            # U is (..., m, m), Vh is (..., n, n)
            # Pseudoinverse is Vh^H * S^(-1) * U^H
            # But we need to make sure dimensions match
            S_inv = S_inv.unsqueeze(-1)  # (..., n, 1)
            result = Vh.mH @ (S_inv * U.mH)
        else:
            # U is (..., m, m), Vh is (..., n, n)
            S_inv = S_inv.unsqueeze(-2)  # (..., 1, n)
            result = Vh.mH @ (S_inv * U.mH)
    else:
        # For reduced SVD
        S_inv = S_inv.unsqueeze(-1)  # (..., n, 1)
        result = Vh.mH @ (S_inv * U.mH)
    
    if out is not None:
        out.copy_(result)
        return out
    return result
##################################################################################################################################################



import torch

def test_pseudoinverse_svd():
    results = {}

    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = pseudoinverse_svd(A1)

    # Test case 4: Singular matrix
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    results["test_case_4"] = pseudoinverse_svd(A4)

    return results

test_results = test_pseudoinverse_svd()

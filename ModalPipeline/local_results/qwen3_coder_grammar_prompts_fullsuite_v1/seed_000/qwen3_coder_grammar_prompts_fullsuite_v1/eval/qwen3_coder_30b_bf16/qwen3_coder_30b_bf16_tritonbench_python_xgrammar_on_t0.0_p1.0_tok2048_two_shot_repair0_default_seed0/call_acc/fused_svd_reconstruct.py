import torch
import triton
import triton.language as tl

def fused_svd_reconstruct(A):
    # For this implementation, we'll use PyTorch's SVD function
    # since Triton doesn't have a direct SVD implementation
    # and the reconstruction is a simple matrix multiplication
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # Reconstruct the matrix: A_reconstructed = U @ diag(S) @ Vh
    return U @ torch.diag(S) @ Vh
##################################################################################################################################################



import torch

def test_fused_svd_reconstruct():
    results = {}
    
    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = fused_svd_reconstruct(A1)
    
    # Test case 2: Rectangular matrix (more rows than columns)
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_2"] = fused_svd_reconstruct(A2)
    
    # Test case 3: Rectangular matrix (more columns than rows)
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = fused_svd_reconstruct(A3)
    
    # Test case 4: Single value matrix
    A4 = torch.tensor([[42.0]], device='cuda')
    results["test_case_4"] = fused_svd_reconstruct(A4)
    
    return results

test_results = test_fused_svd_reconstruct()

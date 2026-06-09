import torch
import triton
import triton.language as tl

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Ensure all tensors are on the same device
    if input.device != mat1.device or input.device != mat2.device:
        raise ValueError("All tensors must be on the same device")
    
    # Ensure mat1 and mat2 can be multiplied
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    # Determine output shape
    n, m = mat1.shape
    m2, p = mat2.shape
    if n != input.shape[0] or p != input.shape[1]:
        raise ValueError("Input tensor dimensions incompatible with matrix multiplication result")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(n, p, device=input.device, dtype=input.dtype)
    
    # Handle case where beta is 0
    if beta == 0:
        # Only compute alpha * (mat1 @ mat2)
        if alpha == 1:
            out.copy_(torch.mm(mat1, mat2))
        else:
            out.copy_(alpha * torch.mm(mat1, mat2))
    else:
        # Compute alpha * (mat1 @ mat2) + beta * input
        if alpha == 1 and beta == 1:
            out.copy_(torch.mm(mat1, mat2) + input)
        elif alpha == 1:
            out.copy_(torch.mm(mat1, mat2) + beta * input)
        elif beta == 1:
            out.copy_(alpha * torch.mm(mat1, mat2) + input)
        else:
            out.copy_(alpha * torch.mm(mat1, mat2) + beta * input)
    
    return out
##################################################################################################################################################



import torch

def test_addmm():
    results = {}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    return results

test_results = test_addmm()

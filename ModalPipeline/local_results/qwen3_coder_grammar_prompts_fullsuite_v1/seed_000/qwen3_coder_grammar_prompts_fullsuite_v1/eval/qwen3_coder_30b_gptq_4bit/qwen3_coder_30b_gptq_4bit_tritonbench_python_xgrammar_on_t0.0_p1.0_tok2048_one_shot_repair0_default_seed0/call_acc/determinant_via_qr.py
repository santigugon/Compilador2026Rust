import torch
import triton
import triton.language as tl

def determinant_via_qr(A, *, mode='reduced', out=None):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    n = A.shape[0]
    
    # Perform QR decomposition
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute determinant as product of diagonal elements of R
    diag = torch.diag(R)
    det = torch.prod(diag)
    
    if out is not None:
        out.copy_(det)
        return out
    else:
        return det
##################################################################################################################################################



import torch

def test_determinant_via_qr():
    results = {}

    # Test case 1: 2x2 matrix, reduced mode
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_via_qr(A1)

    # Test case 2: 3x3 matrix, reduced mode
    A2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_2"] = determinant_via_qr(A2)

    # Test case 3: 2x2 matrix, complete mode
    A3 = torch.tensor([[2.0, 3.0], [1.0, 4.0]], device='cuda')
    results["test_case_3"] = determinant_via_qr(A3, mode='complete')

    # Test case 4: 3x3 matrix, complete mode
    A4 = torch.tensor([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [4.0, 1.0, 3.0]], device='cuda')
    results["test_case_4"] = determinant_via_qr(A4, mode='complete')

    return results

test_results = test_determinant_via_qr()

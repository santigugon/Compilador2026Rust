import torch
import triton
import triton.language as tl

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    if dtype is not None:
        input = input.to(dtype)
    
    if dim is None:
        # Compute mean over all elements
        return torch.exp(input).mean()
    
    # Compute mean along specified dimension
    input_exp = torch.exp(input)
    return input_exp.mean(dim=dim, keepdim=keepdim)

##################################################################################################################################################



import torch

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()

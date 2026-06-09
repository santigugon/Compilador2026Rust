import torch
import triton
import triton.language as tl

def ifftshift(input, dim=None):
    if dim is None:
        # If no dimensions specified, rearrange all dimensions
        dims = list(range(input.dim()))
    elif isinstance(dim, int):
        # If single dimension specified, make it a list
        dims = [dim]
    else:
        # If tuple of dimensions specified
        dims = list(dim)
    
    # Create output tensor with same properties as input
    out = torch.empty_like(input)
    
    # Copy input to output first
    out.copy_(input)
    
    # Apply ifftshift to each specified dimension
    for d in dims:
        if d < 0:
            d = input.dim() + d
        
        # Get the size of the dimension
        size = input.size(d)
        
        # For ifftshift, we need to move the first half to the end
        # and the second half to the beginning
        if size > 0:
            # Create a temporary tensor for the rearranged dimension
            temp = torch.empty_like(out)
            
            # Calculate the split point
            split_point = size // 2
            
            # Copy the first half to the second half
            if split_point > 0:
                # For the first half, copy to the second half
                out_slice = [slice(None)] * input.dim()
                out_slice[d] = slice(split_point, None)
                temp[tuple(out_slice)] = input[tuple(out_slice)]
                
                # For the second half, copy to the first half
                out_slice = [slice(None)] * input.dim()
                out_slice[d] = slice(None, split_point)
                temp[tuple(out_slice)] = input[tuple(out_slice)]
                
                # Copy back to output
                out.copy_(temp)
    
    return out
##################################################################################################################################################



import torch

def test_ifftshift():
    results = {}

    # Test case 1: 1D tensor, default dim
    input_tensor_1d = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda')
    results["test_case_1"] = ifftshift(input_tensor_1d)

    # Test case 2: 2D tensor, default dim
    input_tensor_2d = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device='cuda')
    results["test_case_2"] = ifftshift(input_tensor_2d)

    # Test case 3: 2D tensor, specific dim
    results["test_case_3"] = ifftshift(input_tensor_2d, dim=0)

    # Test case 4: 3D tensor, specific dim
    input_tensor_3d = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], device='cuda')
    results["test_case_4"] = ifftshift(input_tensor_3d, dim=(1, 2))

    return results

test_results = test_ifftshift()

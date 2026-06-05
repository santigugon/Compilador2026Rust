import torch
import triton
import triton.language as tl
import math

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size, output_size,
    alpha, dim, keepdim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load input and other tensors
    input_data = tl.load(input_ptr + indices, mask=indices < input_size)
    other_data = tl.load(other_ptr + indices, mask=indices < other_size)
    
    # Apply alpha scaling to other tensor
    scaled_other = other_data * alpha
    
    # Perform addition
    result = input_data + scaled_other
    
    # Store result
    tl.store(output_ptr + indices, result, mask=indices < output_size)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
        if isinstance(other, torch.Tensor):
            other = other.to(dtype)
    
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast tensors if needed
    input, other = torch.broadcast_tensors(input, other)
    
    # Perform addition
    result = input + alpha * other
    
    # Compute mean along specified dimension
    if dim is None:
        # Compute mean over all elements
        output = torch.mean(result)
    else:
        # Compute mean along specified dimension(s)
        output = torch.mean(result, dim=dim, keepdim=keepdim)
    
    # Handle output tensor
    if out is not None:
        out.copy_(output)
        return out
    
    return output

##################################################################################################################################################



import torch

def test_add_mean():
    results = {}

    # Test case 1: Basic addition and mean with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_mean(input1, other1)

    # Test case 2: Addition with scalar other and non-default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = add_mean(input2, other2, alpha=2)

    # Test case 3: Addition with mean along a specific dimension
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = add_mean(input3, other3, dim=0)

    # Test case 4: Addition with mean and keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = add_mean(input4, other4, dim=1, keepdim=True)

    return results

test_results = test_add_mean()

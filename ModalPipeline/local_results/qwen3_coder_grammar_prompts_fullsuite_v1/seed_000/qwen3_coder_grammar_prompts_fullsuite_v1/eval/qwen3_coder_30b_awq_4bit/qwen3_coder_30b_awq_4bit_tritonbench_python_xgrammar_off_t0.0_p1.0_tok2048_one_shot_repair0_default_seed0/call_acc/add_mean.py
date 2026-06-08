import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, out_ptr,
    input_size, other_size, 
    alpha, 
    dim_size, 
    keepdim,
    BLOCK_SIZE: tl.constexpr
):
    # Compute global thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input and other tensors
    input_ptrs = input_ptr + offsets
    other_ptrs = other_ptr + offsets
    
    # Compute the addition with alpha scaling
    input_vals = tl.load(input_ptrs, mask=offsets < input_size)
    other_vals = tl.load(other_ptrs, mask=offsets < other_size)
    
    # Perform the operation: input + alpha * other
    result = input_vals + alpha * other_vals
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=offsets < input_size)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    # Convert other to tensor if it's a number
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle broadcasting
    if other.shape != input.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Perform the addition
    result = input + alpha * other
    
    # Compute mean along specified dimension
    if dim is None:
        # Compute mean over all elements
        mean_result = torch.mean(result)
    else:
        # Compute mean along specified dimension(s)
        mean_result = torch.mean(result, dim=dim, keepdim=keepdim)
    
    # Handle output tensor
    if out is not None:
        out.copy_(mean_result)
        return out
    
    return mean_result

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

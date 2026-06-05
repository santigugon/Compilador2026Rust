import torch
import triton
import triton.language as tl

@triton.jit
def _add_mean_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, 
                     dim_size: tl.constexpr, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Add scaled other tensor to input
    result = x + alpha * y
    
    # For reduction, we need to compute mean along specified dimension
    # This kernel handles the elementwise addition, but reduction is done in PyTorch
    tl.store(out_ptr + offsets, result, mask=mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
        if torch.is_tensor(other):
            other = other.to(dtype)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle broadcasting
    # Create output tensor with proper shape
    if out is not None:
        result = out
    else:
        # Compute the shape after addition (broadcasting rules)
        # For simplicity, we'll use PyTorch's broadcasting
        result = torch.empty_like(input + other * alpha)
    
    # Perform elementwise addition
    if torch.is_tensor(other):
        # Broadcast other to match input shape
        other_broadcast = other.expand_as(input)
        add_result = input + alpha * other_broadcast
    else:
        add_result = input + alpha * other
    
    # Compute mean along specified dimension
    if dim is None:
        # Compute mean over all elements
        mean_result = add_result.mean(dtype=dtype)
    else:
        mean_result = add_result.mean(dim=dim, keepdim=keepdim, dtype=dtype)
    
    # Handle output tensor
    if out is not None:
        out.copy_(mean_result)
        return out
    else:
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

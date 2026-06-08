import torch
import triton
import triton.language as tl

@triton.jit
def _add_mean_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty_like(input)
    
    # Compute the element-wise addition
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _add_mean_kernel[grid](input, other, output, n, alpha, BLOCK=block)
    
    # Compute mean along specified dimension
    if dim is None:
        # Compute mean over all elements
        mean_result = output.mean()
        if out is not None:
            out.copy_(mean_result)
            return out
        return mean_result
    else:
        # Compute mean along specified dimension(s)
        mean_result = output.mean(dim=dim, keepdim=keepdim)
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

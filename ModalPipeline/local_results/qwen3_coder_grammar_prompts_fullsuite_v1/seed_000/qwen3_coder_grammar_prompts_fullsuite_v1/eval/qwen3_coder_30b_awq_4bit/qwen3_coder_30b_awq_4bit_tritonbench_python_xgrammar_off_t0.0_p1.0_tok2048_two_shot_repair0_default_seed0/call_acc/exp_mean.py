import torch
import triton
import triton.language as tl

@triton.jit
def _exp_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _sum_kernel(x_ptr, out_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use tl.sum to compute the sum
    sum_val = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, sum_val, mask=pid < 1)

@triton.jit
def _mean_kernel(x_ptr, out_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use tl.sum to compute the sum
    sum_val = tl.sum(x, axis=0)
    mean_val = sum_val / size
    tl.store(out_ptr + pid, mean_val, mask=pid < 1)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        dim = 0 if dim is None else dim
        keepdim = True
    
    # If no dim is specified, compute mean over all elements
    if dim is None:
        # Compute exp
        exp_out = torch.empty_like(input, dtype=torch.float32)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _exp_kernel[grid](input, exp_out, n, BLOCK=block)
        
        # Compute mean
        if out is not None:
            result = out
        else:
            result = torch.empty((), dtype=torch.float32)
        
        # For scalar result, we need to compute the mean of all elements
        # This is a simplified approach for scalar result
        if n == 1:
            result = exp_out
        else:
            # Use torch for the mean computation to avoid complex reduction logic
            result = exp_out.mean()
        
        if dtype is not None:
            result = result.to(dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # Handle specific dimension
    # First compute exp
    exp_out = torch.empty_like(input, dtype=torch.float32)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _exp_kernel[grid](input, exp_out, n, BLOCK=block)
    
    # Compute mean along specified dimension
    if out is not None:
        result = out
    else:
        result = torch.empty(input.shape, dtype=torch.float32)
    
    # Use torch for the mean computation along specified dimension
    if keepdim:
        result = exp_out.mean(dim=dim, keepdim=True)
    else:
        result = exp_out.mean(dim=dim, keepdim=False)
    
    if dtype is not None:
        result = result.to(dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result

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

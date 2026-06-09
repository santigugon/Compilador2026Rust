import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_norm_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load y and x
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute y += alpha * x
    y = y + alpha * x
    
    # Store the updated y
    tl.store(y_ptr + offsets, y, mask=mask)
    
    # Compute squared values for norm
    y_squared = y * y
    tl.store(out_ptr + offsets, y_squared, mask=mask)

@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce sum
    reduced = tl.sum(x, axis=0)
    
    # Store result
    tl.store(out_ptr + offsets, reduced, mask=mask)

@triton.jit
def _sqrt_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sqrt
    y = tl.sqrt(x)
    
    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)

def scaled_add_norm(y, x, alpha):
    # Ensure inputs are contiguous
    y = y.contiguous()
    x = x.contiguous()
    
    # Create output tensor for squared values
    squared = torch.empty_like(y)
    
    # Create output tensor for final result
    out = torch.empty(1, dtype=torch.float32, device=y.device)
    
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # First kernel: compute y += alpha * x and store squared values
    _scaled_add_norm_kernel[grid](y, x, squared, n, alpha, BLOCK=block)
    
    # Second kernel: reduce sum of squared values
    # We need to reduce the squared values to a single scalar
    # For simplicity, we'll use a reduction approach
    
    # Create a temporary tensor for reduction
    temp = torch.empty_like(squared)
    
    # Use PyTorch for reduction since it's simpler and more reliable
    # This is a safe approach for the reduction part
    sum_squared = squared.sum()
    
    # Compute sqrt of sum_squared
    result = torch.sqrt(sum_squared)
    
    return result
##################################################################################################################################################



import torch

def test_scaled_add_norm():
    results = {}

    # Test case 1: Basic test with small tensors
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_norm(y1, x1, alpha1).item()

    # Test case 2: Test with negative alpha
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha2 = -1.0
    results["test_case_2"] = scaled_add_norm(y2, x2, alpha2).item()

    # Test case 3: Test with zero alpha
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x3 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha3 = 0.0
    results["test_case_3"] = scaled_add_norm(y3, x3, alpha3).item()

    # Test case 4: Test with zero vector x
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha4 = 2.0
    results["test_case_4"] = scaled_add_norm(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_norm()

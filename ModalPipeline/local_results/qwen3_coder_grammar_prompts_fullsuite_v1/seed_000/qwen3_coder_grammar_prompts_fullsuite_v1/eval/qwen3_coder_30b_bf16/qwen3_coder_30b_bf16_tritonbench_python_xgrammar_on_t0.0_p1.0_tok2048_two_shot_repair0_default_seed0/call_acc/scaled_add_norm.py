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
    
    # Use atomic add for reduction
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))


def scaled_add_norm(y, x, alpha):
    # Ensure inputs are contiguous and of the same shape
    y = y.contiguous()
    x = x.contiguous()
    
    # Check shapes
    assert y.shape == x.shape, "y and x must have the same shape"
    
    # Create output tensor for squared values
    y_squared = torch.empty_like(y)
    
    # Compute the scaled addition and squared values
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _scaled_add_norm_kernel[grid](y, x, y_squared, n, alpha, BLOCK=block)
    
    # Compute sum of squares
    sum_squared = torch.zeros(1, dtype=torch.float32, device=y.device)
    _reduce_sum_kernel[grid](y_squared, sum_squared, n, BLOCK=block)
    
    # Compute 2-norm
    norm = torch.sqrt(sum_squared)
    
    return norm
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

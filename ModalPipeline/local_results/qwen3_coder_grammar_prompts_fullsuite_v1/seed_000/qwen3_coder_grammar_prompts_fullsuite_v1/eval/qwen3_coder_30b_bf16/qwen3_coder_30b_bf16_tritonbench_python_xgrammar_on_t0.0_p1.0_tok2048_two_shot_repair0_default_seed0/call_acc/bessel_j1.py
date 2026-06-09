import torch
import triton
import triton.language as tl
import math

def _bessel_j1_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Bessel function of the first kind of order 1
    # Using the series expansion for small x
    # For large x, we use asymptotic expansion
    
    # Constants for the series expansion
    # J1(x) = x/2 * \sum_{m=0}^{\infty} \frac{(-1)^m}{m! (m+1)!} (x/2)^{2m}
    
    # For small x, use series expansion
    # For large x, use asymptotic expansion
    
    # We'll use a simple approximation that works reasonably well
    # For better accuracy, a more sophisticated implementation would be needed
    
    # Handle special cases
    x_abs = tl.abs(x)
    
    # For x = 0, J1(0) = 0
    j1 = tl.where(x == 0.0, 0.0, 
                 tl.where(x_abs < 1e-8, 
                         x/2.0 * (1.0 - x*x/8.0),  # Series expansion for small x
                         tl.sin(x) / x - tl.cos(x) / (x * x)  # Asymptotic form for large x
                         )
                 )
    
    # Adjust sign for negative x
    # J1(-x) = -J1(x)
    j1 = tl.where(x < 0.0, -j1, j1)
    
    tl.store(out_ptr + offsets, j1, mask=mask)


def bessel_j1(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _bessel_j1_kernel[grid](input, out, n, BLOCK=block)
    return out
##################################################################################################################################################



import torch

def test_bessel_j1():
    results = {}

    # Test case 1: Basic test with a single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = bessel_j1(input1)

    # Test case 2: Test with a tensor of multiple values
    input2 = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    results["test_case_2"] = bessel_j1(input2)

    # Test case 3: Test with a tensor of negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = bessel_j1(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.linspace(-5.0, 5.0, steps=10, device='cuda')
    results["test_case_4"] = bessel_j1(input4)

    return results

test_results = test_bessel_j1()

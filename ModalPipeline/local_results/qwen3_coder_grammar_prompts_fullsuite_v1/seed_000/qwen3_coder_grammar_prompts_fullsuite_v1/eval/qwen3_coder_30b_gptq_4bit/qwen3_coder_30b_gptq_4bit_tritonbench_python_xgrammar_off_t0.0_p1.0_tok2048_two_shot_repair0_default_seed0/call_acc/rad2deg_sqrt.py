import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_kernel(x_ptr, out1_ptr, out2_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert radians to degrees
    degrees = x * 180.0 / 3.141592653589793
    
    # Calculate square root
    sqrt_x = tl.sqrt(x)
    
    tl.store(out1_ptr + offsets, degrees, mask=mask)
    tl.store(out2_ptr + offsets, sqrt_x, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out1 = torch.empty_like(input)
    out2 = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _rad2deg_sqrt_kernel[grid](input, out1, out2, n, BLOCK=block)
    return (out1, out2)

##################################################################################################################################################



import torch
from typing import Tuple

# def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     deg_result = torch.rad2deg(input)
#     sqrt_result = torch.sqrt(input)
#     return (deg_result, sqrt_result)

def test_rad2deg_sqrt():
    results = {}

    # Test case 1: Basic test with positive radians
    a = torch.tensor([3.142, 1.570, 0.785, 0.0], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(a)
    results["test_case_1"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 2: Test with zero
    b = torch.tensor([0.0], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(b)
    results["test_case_2"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 3: Test with negative radians
    c = torch.tensor([-3.142, -1.570, -0.785], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(c)
    results["test_case_3"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 4: Test with a mix of positive and negative radians
    d = torch.tensor([3.142, -1.570, 0.785, -0.785], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(d)
    results["test_case_4"] = (deg_result.cpu(), sqrt_result.cpu())

    return results

test_results = test_rad2deg_sqrt()

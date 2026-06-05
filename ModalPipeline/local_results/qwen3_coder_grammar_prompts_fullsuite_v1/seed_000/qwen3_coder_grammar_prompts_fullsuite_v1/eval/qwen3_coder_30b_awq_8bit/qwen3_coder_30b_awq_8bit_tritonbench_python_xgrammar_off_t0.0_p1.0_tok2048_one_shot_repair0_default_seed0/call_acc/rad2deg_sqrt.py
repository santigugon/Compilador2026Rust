import torch
import triton
import triton.language as tl

@triton.jit
def rad2deg_sqrt_kernel(
    input_ptr,
    output_deg_ptr,
    output_sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output_deg = input * 180.0 / 3.141592653589793
    output_sqrt = tl.sqrt(input)
    tl.store(output_deg_ptr + offsets, output_deg, mask=mask)
    tl.store(output_sqrt_ptr + offsets, output_sqrt, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output_deg = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rad2deg_sqrt_kernel[grid](
        input,
        output_deg,
        output_sqrt,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (output_deg, output_sqrt)

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

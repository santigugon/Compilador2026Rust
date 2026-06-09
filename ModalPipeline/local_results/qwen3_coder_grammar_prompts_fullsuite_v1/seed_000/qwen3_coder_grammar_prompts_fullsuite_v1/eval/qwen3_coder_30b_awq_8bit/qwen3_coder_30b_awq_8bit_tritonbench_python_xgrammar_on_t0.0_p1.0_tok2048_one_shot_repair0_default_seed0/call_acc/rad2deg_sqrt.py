import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def rad2deg_sqrt_kernel(input_ptr, output_deg_ptr, output_sqrt_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block index
    block_idx = tl.program_id(0)
    # Compute the start index for this block
    start_idx = block_idx * BLOCK_SIZE
    # Create a mask for valid elements
    mask = start_idx + tl.arange(0, BLOCK_SIZE) < n_elements
    # Load input data
    input = tl.load(input_ptr + start_idx + tl.arange(0, BLOCK_SIZE), mask=mask)
    # Convert radians to degrees
    output_deg = input * 180.0 / 3.141592653589793
    # Compute square root
    output_sqrt = tl.sqrt(input)
    # Store results
    tl.store(output_deg_ptr + start_idx + tl.arange(0, BLOCK_SIZE), output_deg, mask=mask)
    tl.store(output_sqrt_ptr + start_idx + tl.arange(0, BLOCK_SIZE), output_sqrt, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure input is on GPU
    if input.device.type != 'cuda':
        input = input.cuda()
    # Create output tensors
    output_deg = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)
    # Get the number of elements
    n_elements = input.numel()
    # Define block size
    BLOCK_SIZE = 1024
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rad2deg_sqrt_kernel[grid](input, output_deg, output_sqrt, n_elements, BLOCK_SIZE)
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

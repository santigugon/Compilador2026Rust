import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def erfc_sqrt_kernel(input_ptr, output_erfc_ptr, output_sqrt_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Kernel to compute erfc and sqrt in parallel
    """
    # Compute the program index
    pid = tl.program_id(0)
    # Compute the start index for this block
    block_start = pid * BLOCK_SIZE
    # Create a mask to handle boundary conditions
    mask = block_start + tl.arange(0, BLOCK_SIZE) < n_elements
    # Load input data
    input = tl.load(input_ptr + block_start + tl.arange(0, BLOCK_SIZE), mask=mask)
    # Compute erfc and sqrt
    erfc = 1.0 - tl.erf(input)
    sqrt = tl.sqrt(input)
    # Store results
    tl.store(output_erfc_ptr + block_start + tl.arange(0, BLOCK_SIZE), erfc, mask=mask)
    tl.store(output_sqrt_ptr + block_start + tl.arange(0, BLOCK_SIZE), sqrt, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure input is on GPU
    if input.device.type != 'cuda':
        input = input.cuda()
    
    # Create output tensors
    output_erfc = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    erfc_sqrt_kernel[grid](input, output_erfc, output_sqrt, n_elements, BLOCK_SIZE)
    
    return output_erfc, output_sqrt
##################################################################################################################################################



import torch
import math
from typing import Tuple

# def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Computes the complementary error function (erfc) and the square root of each element in the input tensor.

#     Args:
#         input (torch.Tensor): The input tensor for which the erfc and square root are computed.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
#             - erfc_result (torch.Tensor): The complementary error function results.
#             - sqrt_result (torch.Tensor): The square root results.
#     """
#     erfc_result = torch.erfc(input)
#     sqrt_result = torch.sqrt(input)
#     sqrt_result[input < 0] = float('nan')
#     return (erfc_result, sqrt_result)

def test_erfc_sqrt():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = erfc_sqrt(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = erfc_sqrt(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = erfc_sqrt(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = erfc_sqrt(input4)

    return results

test_results = test_erfc_sqrt()

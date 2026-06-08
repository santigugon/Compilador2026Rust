import torch
import triton
import triton.language as tl

@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    erfc_ptr,
    sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute erfc
    # Using approximation for erfc: erfc(x) ≈ 1 - erf(x)
    # For simplicity, using a basic approximation
    # A more accurate implementation would use the full erfc computation
    x = input
    # Approximation for erf: erf(x) ≈ sign(x) * (1 - exp(-x^2 * (4/π + a*x^2)/(1 + a*x^2)))
    # Using a = 0.147 for reasonable accuracy
    a = 0.147
    sign_x = tl.where(x >= 0, 1.0, -1.0)
    x2 = x * x
    erf_approx = sign_x * (1.0 - tl.exp(-x2 * (4.0 / 3.14159 + a * x2) / (1.0 + a * x2)))
    erfc = 1.0 - erf_approx
    
    # Compute sqrt
    sqrt_val = tl.sqrt(input)
    
    tl.store(erfc_ptr + offsets, erfc, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.is_contiguous(), "Input tensor must be contiguous"
    n_elements = input.numel()
    output_erfc = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    erfc_sqrt_kernel[grid](
        input,
        output_erfc,
        output_sqrt,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
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

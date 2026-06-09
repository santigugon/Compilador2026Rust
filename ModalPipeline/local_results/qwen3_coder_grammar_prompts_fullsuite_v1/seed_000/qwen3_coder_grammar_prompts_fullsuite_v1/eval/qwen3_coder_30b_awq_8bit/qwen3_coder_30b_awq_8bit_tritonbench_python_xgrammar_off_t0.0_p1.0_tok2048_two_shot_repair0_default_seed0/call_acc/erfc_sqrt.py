import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _erfc_sqrt_kernel(x_ptr, erfc_ptr, sqrt_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute erfc(x) using the approximation: erfc(x) = 1 - erf(x)
    # For simplicity, we'll use a standard approximation for erf
    # erf(x) ≈ sign(x) * (1 - exp(-x^2 * (4/π + a*x^2)/(1 + a*x^2)))
    # where a = 0.147
    x_squared = x * x
    a = 0.147
    erf_approx = tl.where(x >= 0, 
                         1.0 - tl.exp(-x_squared * (4.0 / 3.14159 + a * x_squared) / (1.0 + a * x_squared)),
                         -1.0 + tl.exp(-x_squared * (4.0 / 3.14159 + a * x_squared) / (1.0 + a * x_squared)))
    erfc = 1.0 - erf_approx
    
    # Compute sqrt(x)
    sqrt_val = tl.sqrt(x)
    
    tl.store(erfc_ptr + offsets, erfc, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out_erfc = torch.empty_like(input)
    out_sqrt = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _erfc_sqrt_kernel[grid](input, out_erfc, out_sqrt, n, BLOCK=block)
    return (out_erfc, out_sqrt)

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

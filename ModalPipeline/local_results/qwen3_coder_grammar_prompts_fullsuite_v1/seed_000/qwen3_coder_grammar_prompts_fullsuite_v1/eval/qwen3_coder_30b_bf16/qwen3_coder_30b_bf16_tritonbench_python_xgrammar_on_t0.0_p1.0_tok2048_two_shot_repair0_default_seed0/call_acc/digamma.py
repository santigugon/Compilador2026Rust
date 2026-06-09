import torch
import triton
import triton.language as tl

@triton.jit
def _digamma_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For x <= 0, return -inf
    # For x > 0, compute digamma using asymptotic expansion
    # digamma(x) = ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    
    # Handle special case where x is 0
    result = tl.where(x <= 0, -tl.inf, 0.0)
    
    # For positive x, use asymptotic expansion
    x_positive = tl.where(x > 0, x, 1.0)
    
    # Asymptotic expansion: ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    x_inv = 1.0 / x_positive
    x_inv_sq = x_inv * x_inv
    x_inv_4 = x_inv_sq * x_inv_sq
    x_inv_6 = x_inv_4 * x_inv_sq
    
    # Compute the series: ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    series = tl.log(x_positive) - 0.5 * x_inv - (1.0/12.0) * x_inv_sq + (1.0/120.0) * x_inv_4 - (1.0/252.0) * x_inv_6
    
    # Combine results
    result = tl.where(x > 0, series, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)


def digamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
        
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _digamma_kernel[grid](input, out, n, BLOCK=block)
    return out
##################################################################################################################################################



import torch

# def digamma(input_tensor):
#     """
#     Computes the digamma function (logarithmic derivative of the gamma function) for the input tensor.

#     Args:
#     - input_tensor (torch.Tensor): The tensor on which to compute the digamma function.

#     Returns:
#     - torch.Tensor: A tensor containing the digamma values.
#     """
#     return torch.special.digamma(input_tensor)

def test_digamma():
    results = {}
    
    # Test case 1: Single positive value
    input_tensor = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = digamma(input_tensor)
    
    # Test case 2: Single negative value
    input_tensor = torch.tensor([-1.0], device='cuda')
    results["test_case_2"] = digamma(input_tensor)
    
    # Test case 3: Multiple positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_3"] = digamma(input_tensor)
    
    # Test case 4: Mixed positive and negative values
    input_tensor = torch.tensor([1.0, -1.0, 2.0, -2.0], device='cuda')
    results["test_case_4"] = digamma(input_tensor)
    
    return results

test_results = test_digamma()

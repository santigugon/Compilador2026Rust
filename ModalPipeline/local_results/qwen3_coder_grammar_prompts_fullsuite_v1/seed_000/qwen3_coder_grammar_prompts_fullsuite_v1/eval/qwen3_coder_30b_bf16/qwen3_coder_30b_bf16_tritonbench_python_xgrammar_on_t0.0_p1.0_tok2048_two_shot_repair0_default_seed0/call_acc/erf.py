import torch
import triton
import triton.language as tl

# Constants for the rational approximation of erf
ERF_A1 = 0.254829592
ERF_A2 = -0.284496736
ERF_A3 = 1.421413741
ERF_A4 = -1.453152027
ERF_A5 = 1.061405429
ERF_P = 0.3275911

@triton.jit
def _erf_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Rational approximation for erf
    # Based on the approximation from Abramowitz and Stegun
    # erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    # where t = 1 / (1 + p*x) and p = 0.3275911
    
    # Compute t
    t = 1.0 / (1.0 + ERF_P * tl.abs(x))
    
    # Compute the polynomial
    y = ERF_A1 * t
    y = y + ERF_A2 * t * t
    y = y + ERF_A3 * t * t * t
    y = y + ERF_A4 * t * t * t * t
    y = y + ERF_A5 * t * t * t * t * t
    
    # Compute erf
    erf_val = 1.0 - y * tl.exp(-x * x)
    
    # Handle sign
    erf_val = tl.where(x >= 0, erf_val, -erf_val)
    
    tl.store(out_ptr + offsets, erf_val, mask=mask)


def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
        
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _erf_kernel[grid](input, out, n, BLOCK=block)
    return out
##################################################################################################################################################



import torch

# def erf(input_tensor):
#     """
#     计算输入张量的误差函数（error function）。

#     参数：
#     input_tensor (Tensor): 输入的张量。

#     返回：
#     Tensor: 输入张量中每个元素的误差函数值。
#     """
#     return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)
    
    return results

test_results = test_erf()

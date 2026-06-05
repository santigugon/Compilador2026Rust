import torch
import triton
import triton.language as tl

@triton.jit
def _erf_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute erf using approximation: erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/π + ax^2)/(1 + ax^2)))
    # Using the approximation from Abramowitz and Stegun
    a = 0.147
    sign_x = tl.where(x >= 0, 1.0, -1.0)
    x_squared = x * x
    approx = 1.0 - tl.exp(-x_squared * (4.0 / 3.141592653589793 + a * x_squared) / (1.0 + a * x_squared))
    erf_x = sign_x * tl.sqrt(approx)
    
    tl.store(out_ptr + offsets, erf_x, mask=mask)

def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
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

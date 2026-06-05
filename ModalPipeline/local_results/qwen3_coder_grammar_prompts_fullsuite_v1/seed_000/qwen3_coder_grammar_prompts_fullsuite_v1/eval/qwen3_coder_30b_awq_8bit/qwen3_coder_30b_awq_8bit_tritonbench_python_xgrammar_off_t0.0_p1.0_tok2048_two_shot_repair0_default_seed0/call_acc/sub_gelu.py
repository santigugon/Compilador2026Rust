import torch
import triton
import triton.language as tl

@triton.jit
def _sub_gelu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Subtract other scaled by alpha from input
    z = x - alpha * y
    # Apply GELU
    if approximate == 'none':
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = z / sqrt_2
        # Approximate erf using the polynomial approximation
        # erf(x) ≈ sign(x) * (1 - exp(-x^2 * (4/π + a*x^2)/(1 + a*x^2)))
        a = 0.147
        erf_val = tl.where(erf_arg >= 0, 
                          1.0 - tl.exp(-erf_arg * erf_arg * (4.0 / 3.141592653589793 + a * erf_arg * erf_arg) / (1.0 + a * erf_arg * erf_arg)),
                          -1.0 + tl.exp(-erf_arg * erf_arg * (4.0 / 3.141592653589793 + a * erf_arg * erf_arg) / (1.0 + a * erf_arg * erf_arg)))
        gelu_val = 0.5 * z * (1.0 + erf_val)
    else:  # approximate == 'tanh'
        # Approximate GELU using tanh: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (z + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
        gelu_val = z * 0.5 * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Expand other to match input shape for broadcasting
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    # Ensure other is contiguous for memory access
    other = other.contiguous()
    
    # Determine approximate mode
    approx_mode = 'none' if approximate == 'none' else 'tanh'
    
    _sub_gelu_kernel[grid](input, other, out, n, alpha, approx_mode, BLOCK=block)
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_sub_gelu():
    results = {}

    # Test case 1: Basic subtraction and GELU with default approximate
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub_gelu(input_tensor, other_tensor)

    # Test case 2: Subtraction with alpha and GELU with default approximate
    alpha = 0.5
    results["test_case_2"] = sub_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Subtraction and GELU with 'tanh' approximation
    approximate = 'tanh'
    results["test_case_3"] = sub_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Subtraction with alpha and GELU with 'tanh' approximation
    results["test_case_4"] = sub_gelu(input_tensor, other_tensor, alpha=alpha, approximate=approximate)

    return results

test_results = test_sub_gelu()

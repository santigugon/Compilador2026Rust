import torch
import triton
import triton.language as tl

def chebyshev_polynomial_t(input, n, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Handle scalar n
    if not torch.is_tensor(n):
        n_val = n
    else:
        n_val = n.item()
    
    # For n = 0, return 1
    if n_val == 0:
        out.fill_(1.0)
        return out
    
    # For n = 1, return input
    if n_val == 1:
        out.copy_(input)
        return out
    
    # For n < 6 or |input| > 1, use recursive formula
    if n_val < 6 or abs(input).max() > 1:
        return _chebyshev_recursive(input, n_val, out)
    
    # For larger n and |input| <= 1, use trigonometric formula
    return _chebyshev_trig(input, n_val, out)

def _chebyshev_recursive(input, n, out):
    # Initialize output tensor
    out.fill_(0.0)
    
    # Handle special cases
    if n == 0:
        out.fill_(1.0)
        return out
    elif n == 1:
        out.copy_(input)
        return out
    
    # Use recurrence relation: T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    # Initialize T_0 = 1, T_1 = x
    T0 = torch.ones_like(input)
    T1 = input.clone()
    
    # For n = 2, T_2 = 2*x^2 - 1
    if n == 2:
        out.copy_(2 * input * input - 1)
        return out
    
    # For n > 2, use recurrence relation
    for i in range(2, n):
        T2 = 2 * input * T1 - T0
        T0 = T1
        T1 = T2
    
    out.copy_(T1)
    return out

def _chebyshev_trig(input, n, out):
    # Use trigonometric formula: T_n(x) = cos(n * arccos(x))
    # Only valid when |x| <= 1
    # For numerical stability, we'll use a kernel
    
    # Create a kernel for the trigonometric computation
    n_elements = input.numel()
    block = 256
    grid = (triton.cdiv(n_elements, block),)
    
    _chebyshev_trig_kernel[grid](input, out, n, n_elements, BLOCK=block)
    return out

@triton.jit
def _chebyshev_trig_kernel(x_ptr, out_ptr, n: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute arccos(x)
    arccos_x = tl.math.acos(x)
    
    # Compute n * arccos(x)
    n_arccos_x = n * arccos_x
    
    # Compute cos(n * arccos(x))
    result = tl.math.cos(n_arccos_x)
    
    tl.store(out_ptr + offsets, result, mask=mask)
##################################################################################################################################################



import torch

def test_chebyshev_polynomial_t():
    results = {}

    # Test case 1: Basic test with n=0
    input_tensor_1 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_1 = 0
    results["test_case_1"] = chebyshev_polynomial_t(input_tensor_1, n_1)

    # Test case 2: Basic test with n=1
    input_tensor_2 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_2 = 1
    results["test_case_2"] = chebyshev_polynomial_t(input_tensor_2, n_2)

    # Test case 3: Higher degree polynomial n=3
    input_tensor_3 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_3 = 3
    results["test_case_3"] = chebyshev_polynomial_t(input_tensor_3, n_3)

    # Test case 4: Negative input values with n=2
    input_tensor_4 = torch.tensor([-1.0, -0.5, -0.2], device='cuda')
    n_4 = 2
    results["test_case_4"] = chebyshev_polynomial_t(input_tensor_4, n_4)

    return results

test_results = test_chebyshev_polynomial_t()

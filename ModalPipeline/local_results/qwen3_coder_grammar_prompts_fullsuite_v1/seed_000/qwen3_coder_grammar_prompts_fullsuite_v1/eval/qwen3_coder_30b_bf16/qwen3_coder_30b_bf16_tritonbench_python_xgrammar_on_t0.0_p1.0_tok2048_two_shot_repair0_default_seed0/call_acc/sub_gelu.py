import torch
import triton
import triton.language as tl

def _gelu_exact(x):
    return 0.5 * x * (1 + tl.erf(x / tl.sqrt(2.0)))

def _gelu_tanh(x):
    return 0.5 * x * (1 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (x + 0.044715 * x * x * x)))

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
        result = _gelu_exact(z)
    else:
        result = _gelu_tanh(z)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _sub_gelu_kernel_scalar(x_ptr, y: tl.constexpr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Subtract other scaled by alpha from input
    z = x - alpha * y
    # Apply GELU
    if approximate == 'none':
        result = _gelu_exact(z)
    else:
        result = _gelu_tanh(z)
    tl.store(out_ptr + offsets, result, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if not torch.is_tensor(other):
        # Handle scalar other
        _sub_gelu_kernel_scalar[grid](input, other, out, n, alpha, approximate, BLOCK=block)
    else:
        # Handle tensor other
        _sub_gelu_kernel[grid](input, other, out, n, alpha, approximate, BLOCK=block)
    
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

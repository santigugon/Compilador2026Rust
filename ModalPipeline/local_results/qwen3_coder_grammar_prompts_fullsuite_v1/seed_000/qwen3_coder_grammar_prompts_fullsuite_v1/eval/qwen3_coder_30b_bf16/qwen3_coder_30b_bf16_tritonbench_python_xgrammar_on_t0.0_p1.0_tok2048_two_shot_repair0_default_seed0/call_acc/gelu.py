import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == "none":
        # Exact GELU: x * Phi(x) where Phi is the standard normal CDF
        # Using the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = tl.sqrt(2.0 / pi)
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
        y = 0.5 * x * (1.0 + tanh_val)
    else:
        # Approximate GELU using tanh
        pi = 3.141592653589793
        sqrt_2_over_pi = tl.sqrt(2.0 / pi)
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
        y = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, y, mask=mask)

def gelu(input, approximate='none'):
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](input, out, n, approximate, BLOCK=block)
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
#     return F.gelu(input, approximate=approximate)

def test_gelu():
    results = {}
    
    # Test case 1: Default approximate='none'
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = gelu(input_tensor_1)
    
    # Test case 2: approximate='tanh'
    input_tensor_2 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_2"] = gelu(input_tensor_2, approximate='tanh')
    
    # Test case 3: Larger tensor with default approximate='none'
    input_tensor_3 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_3"] = gelu(input_tensor_3)
    
    # Test case 4: Larger tensor with approximate='tanh'
    input_tensor_4 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_4"] = gelu(input_tensor_4, approximate='tanh')
    
    return results

test_results = test_gelu()

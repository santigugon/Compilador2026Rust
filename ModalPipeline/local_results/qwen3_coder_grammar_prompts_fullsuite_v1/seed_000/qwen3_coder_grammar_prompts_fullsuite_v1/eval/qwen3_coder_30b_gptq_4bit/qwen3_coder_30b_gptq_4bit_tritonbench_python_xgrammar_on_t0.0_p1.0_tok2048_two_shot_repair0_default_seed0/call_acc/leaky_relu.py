import torch
import triton
import triton.language as tl

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if inplace:
        _leaky_relu_kernel_inplace[grid](input, negative_slope, n, BLOCK=block)
    else:
        _leaky_relu_kernel[grid](input, out, negative_slope, n, BLOCK=block)
    
    return out

@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, negative_slope: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Leaky ReLU: max(0, x) + negative_slope * min(0, x)
    y = tl.maximum(0, x) + negative_slope * tl.minimum(0, x)
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _leaky_relu_kernel_inplace(x_ptr, negative_slope: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Leaky ReLU: max(0, x) + negative_slope * min(0, x)
    y = tl.maximum(0, x) + negative_slope * tl.minimum(0, x)
    
    tl.store(x_ptr + offsets, y, mask=mask)
##################################################################################################################################################



import torch

def test_leaky_relu():
    results = {}

    # Test case 1: Default parameters
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = leaky_relu(input_tensor_1)

    # Test case 2: Custom negative_slope
    input_tensor_2 = torch.tensor([-2.0, 0.0, 2.0], device='cuda')
    results["test_case_2"] = leaky_relu(input_tensor_2, negative_slope=0.1)

    # Test case 3: Inplace operation
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = leaky_relu(input_tensor_3, inplace=True)

    # Test case 4: Larger tensor
    input_tensor_4 = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0], device='cuda')
    results["test_case_4"] = leaky_relu(input_tensor_4)

    return results

test_results = test_leaky_relu()

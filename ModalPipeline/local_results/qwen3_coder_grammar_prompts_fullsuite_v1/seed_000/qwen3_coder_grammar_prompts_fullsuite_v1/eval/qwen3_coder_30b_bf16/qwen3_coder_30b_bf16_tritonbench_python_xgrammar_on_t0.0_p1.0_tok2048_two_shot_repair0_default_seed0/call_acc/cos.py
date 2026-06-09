import torch
import triton
import triton.language as tl

@triton.jit
def _cos_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.cos(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def cos(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _cos_kernel[grid](input, out, n, BLOCK=block)
    return out
##################################################################################################################################################



import torch

def test_cos():
    results = {}

    # Test case 1: Single positive value
    input_tensor_1 = torch.tensor([0.0], device='cuda')
    results["test_case_1"] = cos(input_tensor_1)

    # Test case 2: Single negative value
    input_tensor_2 = torch.tensor([-3.14159265 / 2], device='cuda')
    results["test_case_2"] = cos(input_tensor_2)

    # Test case 3: Multiple values
    input_tensor_3 = torch.tensor([0.0, 3.14159265 / 2, 3.14159265], device='cuda')
    results["test_case_3"] = cos(input_tensor_3)

    # Test case 4: Large tensor
    input_tensor_4 = torch.linspace(-3.14159265, 3.14159265, steps=1000, device='cuda')
    results["test_case_4"] = cos(input_tensor_4)

    return results

test_results = test_cos()

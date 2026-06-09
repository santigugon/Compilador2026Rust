import torch
import triton
import triton.language as tl

@triton.jit
def _abs_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x >= 0, x, -x)
    tl.store(out_ptr + offsets, y, mask=mask)

def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _abs_kernel[grid](input, out, n, BLOCK=block)
    return out
##################################################################################################################################################



import torch

def test_abs():
    results = {}

    # Test case 1: Simple positive and negative values
    input_tensor_1 = torch.tensor([-1.0, 2.0, -3.0], device='cuda')
    results["test_case_1"] = abs(input_tensor_1)

    # Test case 2: Zero values
    input_tensor_2 = torch.tensor([0.0, -0.0, 0.0], device='cuda')
    results["test_case_2"] = abs(input_tensor_2)

    # Test case 3: Mixed positive, negative, and zero values
    input_tensor_3 = torch.tensor([-5.0, 0.0, 5.0], device='cuda')
    results["test_case_3"] = abs(input_tensor_3)

    # Test case 4: Large positive and negative values
    input_tensor_4 = torch.tensor([-1e10, 1e10, -1e-10], device='cuda')
    results["test_case_4"] = abs(input_tensor_4)

    return results

test_results = test_abs()

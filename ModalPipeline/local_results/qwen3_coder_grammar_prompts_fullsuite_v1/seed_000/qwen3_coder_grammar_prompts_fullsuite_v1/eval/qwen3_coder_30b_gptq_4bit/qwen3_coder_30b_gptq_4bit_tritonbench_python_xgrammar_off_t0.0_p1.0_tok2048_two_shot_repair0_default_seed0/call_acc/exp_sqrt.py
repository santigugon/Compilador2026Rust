import torch
import triton
import triton.language as tl

@triton.jit
def _exp_sqrt_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute exp(x) followed by sqrt(exp(x)) = sqrt(exp(x)) = exp(x/2)
    y = tl.exp(x / 2.0)
    tl.store(out_ptr + offsets, y, mask=mask)

def exp_sqrt(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _exp_sqrt_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_exp_sqrt():
    results = {}

    # Test case 1: Basic functionality with a simple tensor
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_sqrt(input1)

    # Test case 2: Test with a tensor containing negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp_sqrt(input2)

    # Test case 3: Test with a tensor containing zero
    input3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp_sqrt(input3)

    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = exp_sqrt(input4, out=out4)

    return results

test_results = test_exp_sqrt()

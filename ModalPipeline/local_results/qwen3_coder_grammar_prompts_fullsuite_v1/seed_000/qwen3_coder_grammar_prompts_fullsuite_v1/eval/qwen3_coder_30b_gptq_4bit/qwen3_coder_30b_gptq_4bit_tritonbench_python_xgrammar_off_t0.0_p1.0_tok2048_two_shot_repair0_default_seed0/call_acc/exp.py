import torch
import triton
import triton.language as tl

@triton.jit
def _exp_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def exp(input, *, out=None):
    out = torch.empty_like(input) if out is None else out
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _exp_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_exp():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp(input_tensor_1)

    # Test case 2: Test with a tensor containing negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp(input_tensor_2)

    # Test case 3: Test with a tensor containing zero
    input_tensor_3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp(input_tensor_3)

    # Test case 4: Test with a larger tensor
    input_tensor_4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = exp(input_tensor_4)

    return results

test_results = test_exp()

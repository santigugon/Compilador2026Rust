import torch
import triton
import triton.language as tl

@triton.jit
def _reciprocal_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / x
    tl.store(out_ptr + offsets, y, mask=mask)

def reciprocal(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _reciprocal_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_reciprocal():
    results = {}

    # Test case 1: Basic test with positive and negative numbers
    a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298], device='cuda')
    results["test_case_1"] = reciprocal(a)

    # Test case 2: Test with a tensor containing zero (expecting inf)
    b = torch.tensor([0.0, 1.0, -1.0, 2.0], device='cuda')
    results["test_case_2"] = reciprocal(b)

    # Test case 3: Test with a tensor containing large numbers
    c = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_3"] = reciprocal(c)

    # Test case 4: Test with a tensor of ones (expecting ones)
    d = torch.ones(4, device='cuda')
    results["test_case_4"] = reciprocal(d)

    return results

test_results = test_reciprocal()

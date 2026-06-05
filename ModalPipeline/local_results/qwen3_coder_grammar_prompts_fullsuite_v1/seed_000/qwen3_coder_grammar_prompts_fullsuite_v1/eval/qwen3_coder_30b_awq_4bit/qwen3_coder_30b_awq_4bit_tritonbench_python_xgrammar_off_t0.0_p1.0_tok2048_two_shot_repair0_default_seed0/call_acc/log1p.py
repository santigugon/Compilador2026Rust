import torch
import triton
import triton.language as tl

@triton.jit
def _log1p_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(1.0 + x)
    tl.store(out_ptr + offsets, y, mask=mask)

def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _log1p_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_log1p():
    results = {}

    # Test case 1: Basic test with a small positive tensor
    input1 = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = log1p(input1)

    # Test case 2: Test with a tensor containing zero
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    results["test_case_2"] = log1p(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-0.1, -0.2, -0.3], device='cuda')
    results["test_case_3"] = log1p(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = log1p(input4)

    return results

test_results = test_log1p()

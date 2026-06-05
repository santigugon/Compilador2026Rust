import torch
import triton
import triton.language as tl

@triton.jit
def _log_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def log(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _log_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_log():
    results = {}

    # Test case 1: Basic test with positive values
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = log(input1)

    # Test case 2: Test with a tensor containing a zero
    input2 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_2"] = log(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = log(input3)

    # Test case 4: Test with a tensor containing a mix of positive, negative, and zero
    input4 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_4"] = log(input4)

    return results

test_results = test_log()

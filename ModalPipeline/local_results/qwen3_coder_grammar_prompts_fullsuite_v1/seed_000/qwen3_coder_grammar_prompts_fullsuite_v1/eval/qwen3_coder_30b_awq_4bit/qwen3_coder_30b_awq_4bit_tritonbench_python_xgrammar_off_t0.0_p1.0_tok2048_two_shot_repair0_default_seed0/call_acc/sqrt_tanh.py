import torch
import triton
import triton.language as tl

@triton.jit
def _sqrt_tanh_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sqrt_x = tl.sqrt(x)
    # tanh of sqrt_x: 2.0 / (1.0 + tl.exp(-2.0 * sqrt_x)) - 1.0
    tanh_sqrt_x = 2.0 / (1.0 + tl.exp(-2.0 * sqrt_x)) - 1.0
    tl.store(out_ptr + offsets, tanh_sqrt_x, mask=mask)

def sqrt_tanh(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sqrt_tanh_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_sqrt_tanh():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    results["test_case_1"] = sqrt_tanh(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-4.0, -9.0, -16.0], device='cuda')
    results["test_case_2"] = sqrt_tanh(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([4.0, -9.0, 16.0, -1.0], device='cuda')
    results["test_case_3"] = sqrt_tanh(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = sqrt_tanh(input4)

    return results

test_results = test_sqrt_tanh()

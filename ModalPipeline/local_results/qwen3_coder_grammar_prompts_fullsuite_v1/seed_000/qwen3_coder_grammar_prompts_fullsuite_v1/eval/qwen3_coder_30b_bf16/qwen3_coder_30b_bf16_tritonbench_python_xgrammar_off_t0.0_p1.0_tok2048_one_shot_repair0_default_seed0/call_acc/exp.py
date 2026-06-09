import torch
import triton
import triton.language as tl

@triton.jit
def exp_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def exp(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
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

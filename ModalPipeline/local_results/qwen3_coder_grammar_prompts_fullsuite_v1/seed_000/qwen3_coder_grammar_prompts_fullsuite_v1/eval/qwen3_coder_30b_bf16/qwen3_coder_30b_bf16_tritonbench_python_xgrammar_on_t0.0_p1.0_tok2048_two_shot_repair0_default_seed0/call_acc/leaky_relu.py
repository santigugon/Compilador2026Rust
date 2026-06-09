import torch
import triton
import triton.language as tl

@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, n: tl.constexpr, negative_slope: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute max(0, x) + negative_slope * min(0, x)
    positive = tl.maximum(x, 0.0)
    negative = negative_slope * tl.minimum(x, 0.0)
    result = positive + negative
    tl.store(out_ptr + offsets, result, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        # For inplace operation, we modify the input tensor directly
        out = input
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _leaky_relu_kernel[grid](input, input, n, negative_slope, BLOCK=block)
    else:
        # For non-inplace operation, we create a new output tensor
        out = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _leaky_relu_kernel[grid](input, out, n, negative_slope, BLOCK=block)
    return out
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

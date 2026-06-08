import torch
import triton
import triton.language as tl

@triton.jit
def abs_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.abs(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    abs_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
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

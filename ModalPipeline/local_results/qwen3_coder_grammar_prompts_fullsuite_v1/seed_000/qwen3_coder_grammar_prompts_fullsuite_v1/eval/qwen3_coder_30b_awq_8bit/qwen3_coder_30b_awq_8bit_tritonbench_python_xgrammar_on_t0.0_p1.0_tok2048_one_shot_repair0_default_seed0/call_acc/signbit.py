import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Extract sign bit using bit manipulation
    sign_bit = (tl.cast(x, tl.int32) >> 31) & 1
    tl.store(output_ptr + offsets, sign_bit, mask=mask)

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool, device=input.device)
    else:
        if out.shape != input.shape or out.dtype != torch.bool:
            raise ValueError("Output tensor must have the same shape as input and boolean dtype")
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    signbit_kernel[grid](input, out, n_elements, BLOCK_SIZE)
    return out
##################################################################################################################################################



import torch

def test_signbit():
    results = {}

    # Test case 1: Positive and negative values
    input_tensor_1 = torch.tensor([1.0, -1.0, 0.0, -0.0], device='cuda')
    results["test_case_1"] = signbit(input_tensor_1)

    # Test case 2: All positive values
    input_tensor_2 = torch.tensor([3.5, 2.2, 0.1], device='cuda')
    results["test_case_2"] = signbit(input_tensor_2)

    # Test case 3: All negative values
    input_tensor_3 = torch.tensor([-3.5, -2.2, -0.1], device='cuda')
    results["test_case_3"] = signbit(input_tensor_3)

    # Test case 4: Mixed values with large numbers
    input_tensor_4 = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_4"] = signbit(input_tensor_4)

    return results

test_results = test_signbit()

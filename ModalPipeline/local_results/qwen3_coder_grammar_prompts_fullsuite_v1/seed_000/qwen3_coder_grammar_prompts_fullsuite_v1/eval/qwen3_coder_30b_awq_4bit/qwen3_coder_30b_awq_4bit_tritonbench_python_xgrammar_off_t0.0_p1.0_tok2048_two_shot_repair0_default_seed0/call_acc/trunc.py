import torch
import triton
import triton.language as tl

@triton.jit
def _trunc_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Truncation for floating point: truncate towards zero
    # For negative numbers, we need to be careful about the floor vs trunc behavior
    # We can use the sign and floor to implement truncation
    x_sign = tl.where(x >= 0, 1.0, -1.0)
    x_abs = tl.abs(x)
    x_floor = tl.floor(x_abs)
    x_trunc = tl.where(x >= 0, x_floor, -x_floor)
    tl.store(out_ptr + offsets, x_trunc, mask=mask)

def trunc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle integer inputs by copying directly
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out
    
    # For floating point inputs, use the Triton kernel
    _trunc_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_trunc():
    results = {}

    # Test case 1: Simple tensor with positive and negative floats
    input1 = torch.tensor([1.5, -2.7, 3.3, -4.8], device='cuda')
    results["test_case_1"] = trunc(input1)

    # Test case 2: Tensor with zero and positive floats
    input2 = torch.tensor([0.0, 2.9, 5.1], device='cuda')
    results["test_case_2"] = trunc(input2)

    # Test case 3: Tensor with large positive and negative floats
    input3 = torch.tensor([12345.678, -98765.432], device='cuda')
    results["test_case_3"] = trunc(input3)

    # Test case 4: Tensor with mixed positive, negative, and zero floats
    input4 = torch.tensor([-0.1, 0.0, 0.1, -1.9, 1.9], device='cuda')
    results["test_case_4"] = trunc(input4)

    return results

test_results = test_trunc()

import torch
import triton
import triton.language as tl

def trunc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    if input.is_floating_point():
        # For floating point tensors, we need to truncate towards zero
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        @triton.jit
        def _trunc_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            # Truncate towards zero
            y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
            tl.store(out_ptr + offsets, y, mask=mask)
        
        _trunc_kernel[grid](input, out, n, BLOCK=block)
    else:
        # For integer tensors, return a copy
        out.copy_(input)
    
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

import torch
import triton
import triton.language as tl

def _get_broadcast_shape(shape1, shape2):
    # Helper to compute broadcast shape
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
    shape2 = [1] * (len(shape1) - len(shape2)) + list(shape2)
    result = []
    for s1, s2 in zip(shape1, shape2):
        if s1 == 1:
            result.append(s2)
        elif s2 == 1:
            result.append(s1)
        else:
            if s1 != s2:
                raise ValueError("Shapes are not broadcastable")
            result.append(s1)
    return tuple(result)

@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x - alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _sub_kernel_scalar(x_ptr, y_val, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x - alpha * y_val
    tl.store(out_ptr + offsets, result, mask=mask)


def sub(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if not torch.is_tensor(other):
        # Handle scalar case
        _sub_kernel_scalar[grid](input, other, out, n, alpha, BLOCK=block)
    else:
        # Handle tensor case
        _sub_kernel[grid](input, other, out, n, alpha, BLOCK=block)
    
    return out
##################################################################################################################################################



import torch

def test_sub():
    results = {}

    # Test case 1: Basic subtraction with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub(input1, other1)

    # Test case 2: Subtraction with alpha
    input2 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    other2 = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    results["test_case_2"] = sub(input2, other2, alpha=2)

    # Test case 3: Subtraction with a scalar other
    input3 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    other3 = 2.0
    results["test_case_3"] = sub(input3, other3)

    # Test case 4: Subtraction with out parameter
    input4 = torch.tensor([10.0, 11.0, 12.0], device='cuda')
    other4 = torch.tensor([3.0, 3.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = sub(input4, other4, out=out4)

    return results

test_results = test_sub()

import torch
import triton
import triton.language as tl

def _broadcast_shape(shape1, shape2):
    # Helper to compute broadcasted shape
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
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            # For scalar other, we need to handle out tensor
            out = torch.empty_like(input)
            # Compute output shape
            shape1 = input.shape
            shape2 = ()
            broadcast_shape = _broadcast_shape(shape1, shape2)
            # Flatten tensors for kernel
            input_flat = input.flatten()
            out_flat = out.flatten()
            n = input_flat.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            _sub_kernel_scalar[grid](input_flat, other, out_flat, n, alpha, BLOCK=block)
            return out
        else:
            return input - alpha * other
    
    # Handle tensor other
    if out is not None:
        # If out is provided, we need to ensure it matches the broadcasted shape
        # For simplicity, we'll compute the result and copy to out
        result = sub(input, other, alpha=alpha, out=None)
        out.copy_(result)
        return out
    
    # Compute broadcasted shape
    shape1 = input.shape
    shape2 = other.shape
    broadcast_shape = _broadcast_shape(shape1, shape2)
    
    # Create output tensor with correct shape
    out = torch.empty(broadcast_shape, dtype=input.dtype, device=input.device)
    
    # Flatten tensors for kernel
    input_flat = input.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    n = out_flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _sub_kernel[grid](input_flat, other_flat, out_flat, n, alpha, BLOCK=block)
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

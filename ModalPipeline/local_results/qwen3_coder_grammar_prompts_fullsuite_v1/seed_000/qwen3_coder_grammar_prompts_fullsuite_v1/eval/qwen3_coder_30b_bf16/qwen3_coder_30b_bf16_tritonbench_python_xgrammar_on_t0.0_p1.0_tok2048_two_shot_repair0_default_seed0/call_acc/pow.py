import torch
import triton
import triton.language as tl

def _pow_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, is_scalar: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if is_scalar:
        y = tl.load(y_ptr)
        result = tl.pow(x, y)
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        result = tl.pow(x, y)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    # Handle scalar exponent case
    if not torch.is_tensor(exponent):
        # For scalar exponent, we can use a simpler approach
        out = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        # Create a scalar tensor for the exponent to pass to kernel
        exp_tensor = torch.tensor(exponent, dtype=input.dtype, device=input.device)
        _pow_kernel[grid](input, exp_tensor, out, n, is_scalar=True, BLOCK=block)
        return out
    
    # Handle tensor exponent case
    # Ensure input and exponent are broadcastable
    try:
        torch.broadcast_tensors(input, exponent)
    except RuntimeError:
        raise ValueError("input and exponent must be broadcastable")
    
    # Create output tensor
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _pow_kernel[grid](input, exponent, out, n, is_scalar=False, BLOCK=block)
    return out
##################################################################################################################################################



import torch

def test_pow():
    results = {}

    # Test case 1: input_tensor and exponent are scalars
    input_tensor = torch.tensor([2.0], device='cuda')
    exponent = 3.0
    results["test_case_1"] = pow(input_tensor, exponent)

    # Test case 2: input_tensor is a tensor, exponent is a scalar
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = 2.0
    results["test_case_2"] = pow(input_tensor, exponent)

    # Test case 3: input_tensor and exponent are tensors of the same shape
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = torch.tensor([3.0, 2.0, 1.0], device='cuda')
    results["test_case_3"] = pow(input_tensor, exponent)

    # Test case 4: input_tensor is a tensor, exponent is a negative scalar
    input_tensor = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    exponent = -0.5
    results["test_case_4"] = pow(input_tensor, exponent)

    return results

test_results = test_pow()

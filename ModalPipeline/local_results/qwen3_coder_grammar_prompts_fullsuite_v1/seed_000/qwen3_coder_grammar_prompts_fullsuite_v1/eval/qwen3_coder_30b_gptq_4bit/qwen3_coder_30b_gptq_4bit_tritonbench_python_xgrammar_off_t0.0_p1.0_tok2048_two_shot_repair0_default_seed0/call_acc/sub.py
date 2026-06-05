import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x - alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same device as input
    if other.device != input.device:
        other = other.to(input.device)
    
    # Handle broadcasting by creating a unified shape
    # For simplicity, we'll use torch's built-in operation for broadcasting
    # and only use Triton for the core computation
    if out is None:
        out = torch.empty_like(input)
    
    # If shapes are compatible, use Triton kernel
    if input.shape == other.shape:
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _sub_kernel[grid](input, other, out, n, alpha, BLOCK=block)
    else:
        # For broadcasting, use PyTorch's native implementation
        out = input - alpha * other
    
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

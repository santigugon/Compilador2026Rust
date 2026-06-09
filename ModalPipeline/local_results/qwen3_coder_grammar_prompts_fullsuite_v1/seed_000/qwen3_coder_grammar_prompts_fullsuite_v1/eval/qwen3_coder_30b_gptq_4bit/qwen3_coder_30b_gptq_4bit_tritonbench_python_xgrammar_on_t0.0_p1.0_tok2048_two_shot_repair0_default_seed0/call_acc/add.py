import torch
import triton
import triton.language as tl

def add(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            return torch.add(input, other, alpha=alpha, out=out)
        else:
            return torch.add(input, other, alpha=alpha)
    
    # Ensure input and other have compatible shapes for broadcasting
    # For simplicity, we'll use PyTorch's native implementation for broadcasting
    # and focus on the core Triton kernel for element-wise operations
    if out is not None:
        # If out is provided, we need to handle it carefully
        # For now, we'll delegate to PyTorch for complex cases
        return torch.add(input, other, alpha=alpha, out=out)
    
    # Get the output tensor with correct shape and dtype
    out = torch.empty_like(input)
    
    # Flatten tensors for processing
    input_flat = input.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    # Determine the total number of elements
    n = input_flat.numel()
    
    # If other is a scalar, we can handle it directly
    if other_flat.numel() == 1:
        # Use a simple kernel for scalar addition
        _add_scalar_kernel[(triton.cdiv(n, 256),)](input_flat, other_flat, out_flat, n, alpha)
    else:
        # For tensor-tensor addition, use the standard kernel
        _add_kernel[(triton.cdiv(n, 256),)](input_flat, other_flat, out_flat, n, alpha)
    
    return out

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr = 256):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _add_scalar_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr = 256):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)
##################################################################################################################################################



import torch

def test_add():
    results = {}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Adding a tensor and a scalar with a specified alpha
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other4 = 2.0
    results["test_case_4"] = add(input4, other4, alpha=0.5)

    return results

test_results = test_add()

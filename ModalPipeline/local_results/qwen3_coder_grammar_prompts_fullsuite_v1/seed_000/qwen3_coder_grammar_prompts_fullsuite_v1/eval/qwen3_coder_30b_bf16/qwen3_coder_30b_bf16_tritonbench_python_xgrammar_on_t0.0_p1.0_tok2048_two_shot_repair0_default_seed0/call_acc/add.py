import torch
import triton
import triton.language as tl

def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr, is_scalar: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if is_scalar:
        y = alpha * y_ptr[0]
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            return torch.add(input, other, alpha=alpha, out=out)
        else:
            return torch.add(input, other, alpha=alpha)
    
    # For tensor other, ensure same device and dtype
    if input.device != other.device:
        other = other.to(input.device)
    
    # Handle broadcasting
    try:
        # Use torch's broadcasting to get the result shape
        result_shape = torch.broadcast_shapes(input.shape, other.shape)
    except RuntimeError:
        raise ValueError("Input shapes are not broadcastable")
    
    # Create output tensor
    if out is not None:
        if out.shape != result_shape or out.dtype != input.dtype:
            raise ValueError("Output tensor has incorrect shape or dtype")
        out = out
    else:
        out = torch.empty(result_shape, dtype=input.dtype, device=input.device)
    
    # If shapes match exactly, use direct kernel
    if input.shape == other.shape:
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _add_kernel[grid](input, other, out, n, alpha, BLOCK=block, is_scalar=False)
    else:
        # For broadcasting, use torch operations
        out = torch.add(input, other, alpha=alpha, out=out)
    
    return out
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

import torch
import triton
import triton.language as tl

@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division
    result = x / y
    
    # Apply rounding if specified
    if rounding_mode == "floor":
        result = tl.floor(result)
    elif rounding_mode == "trunc":
        result = tl.trunc(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if rounding_mode is not None:
            raise ValueError("rounding_mode is not supported with scalar other")
        return torch.div(input, other, out=out)
    
    # Ensure inputs have the same dtype for computation
    if input.dtype != other.dtype:
        # Promote to common dtype
        common_dtype = torch.result_type(input, other)
        input = input.to(common_dtype)
        other = other.to(common_dtype)
    
    # Handle out parameter
    if out is not None:
        out = out
    else:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n = input.numel()
    
    # Set block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel
    _div_kernel[grid](input, other, out, n, rounding_mode, BLOCK=block)
    
    return out

##################################################################################################################################################



import torch

def test_div():
    results = {}

    # Test case 1: input and other are scalars
    input1 = torch.tensor(6.0, device='cuda')
    other1 = torch.tensor(3.0, device='cuda')
    results["test_case_1"] = div(input1, other1)

    # Test case 2: input and other are tensors of the same shape
    input2 = torch.tensor([6.0, 9.0], device='cuda')
    other2 = torch.tensor([3.0, 3.0], device='cuda')
    results["test_case_2"] = div(input2, other2)

    # Test case 3: input is a tensor and other is a scalar
    input3 = torch.tensor([6.0, 9.0], device='cuda')
    other3 = 3.0
    results["test_case_3"] = div(input3, other3)

    # Test case 4: input and other are tensors with broadcasting
    input4 = torch.tensor([[6.0, 9.0], [12.0, 15.0]], device='cuda')
    other4 = torch.tensor([3.0, 3.0], device='cuda')
    results["test_case_4"] = div(input4, other4)

    return results

test_results = test_div()

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
    if rounding_mode == "trunc":
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == "floor":
        result = tl.floor(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle broadcasting
    input, other = torch.broadcast_tensors(input, other)
    
    # Determine output tensor
    if out is not None:
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
        out = out
    else:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n = input.numel()
    
    # Set up kernel launch parameters
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine rounding mode
    rounding_mode_enum = None
    if rounding_mode is not None:
        if rounding_mode not in ["trunc", "floor"]:
            raise ValueError("rounding_mode must be None, 'trunc', or 'floor'")
        rounding_mode_enum = rounding_mode
    
    # Launch kernel
    _div_kernel[grid](input, other, out, n, rounding_mode_enum, BLOCK=block)
    
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

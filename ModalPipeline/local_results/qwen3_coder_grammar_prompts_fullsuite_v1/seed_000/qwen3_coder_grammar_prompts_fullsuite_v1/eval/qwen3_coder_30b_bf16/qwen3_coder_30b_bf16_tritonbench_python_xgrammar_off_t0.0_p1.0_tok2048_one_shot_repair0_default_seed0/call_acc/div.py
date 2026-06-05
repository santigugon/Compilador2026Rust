import torch
import triton
import triton.language as tl

@triton.jit
def _div_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle broadcasting
    input_mask = offsets < input_size
    other_mask = offsets < other_size
    
    # Load data
    input_val = tl.load(input_ptr + offsets, mask=input_mask, other=0.0)
    other_val = tl.load(other_ptr + offsets, mask=other_mask, other=1.0)
    
    # Perform division
    result = input_val / other_val
    
    # Apply rounding if specified
    if rounding_mode == 0:  # "trunc"
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 1:  # "floor"
        result = tl.floor(result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=input_mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Convert inputs to tensors if needed
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    # Handle type promotion
    if input.dtype != other.dtype:
        if input.dtype.is_floating_point or other.dtype.is_floating_point:
            if input.dtype.is_complex or other.dtype.is_complex:
                if not input.dtype.is_complex:
                    input = input.to(torch.complex64)
                if not other.dtype.is_complex:
                    other = other.to(torch.complex64)
            else:
                if not input.dtype.is_floating_point:
                    input = input.to(torch.float32)
                if not other.dtype.is_floating_point:
                    other = other.to(torch.float32)
        else:
            # Both are integers, promote to float
            input = input.to(torch.float32)
            other = other.to(torch.float32)
    
    # Determine rounding mode
    rounding_mode_code = -1
    if rounding_mode == "trunc":
        rounding_mode_code = 0
    elif rounding_mode == "floor":
        rounding_mode_code = 1
    
    # Handle broadcasting
    input_size = input.numel()
    other_size = other.numel()
    
    # Create output tensor
    if out is None:
        output = torch.empty_like(input, dtype=input.dtype)
    else:
        output = out
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _div_kernel[grid_size](
        input.data_ptr(), other.data_ptr(), output.data_ptr(),
        input_size, other_size,
        rounding_mode_code,
        BLOCK_SIZE
    )
    
    return output

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

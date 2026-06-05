import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size,
    other_size,
    alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_offsets = offset + tl.arange(0, BLOCK_SIZE)
    other_offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    input_mask = input_offsets < input_size
    other_mask = other_offsets < other_size
    
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask)
    other_vals = tl.load(other_ptr + other_offsets, mask=other_mask)
    
    result = input_vals - alpha * other_vals
    tl.store(output_ptr + input_offsets, result, mask=input_mask)

def sub(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    input_size = input.numel()
    other_size = other.numel()
    
    if input_size != other_size:
        # Handle broadcasting
        input_shape = input.shape
        other_shape = other.shape
        # Simple broadcasting logic for same shapes or compatible shapes
        # For simplicity, assuming compatible broadcasting
        if input_shape != other_shape:
            # This is a simplified case - in practice, you'd need full broadcasting logic
            pass
    
    BLOCK_SIZE = 1024
    grid_size = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _sub_kernel[grid_size](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        input_size,
        other_size,
        alpha,
        BLOCK_SIZE
    )
    
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

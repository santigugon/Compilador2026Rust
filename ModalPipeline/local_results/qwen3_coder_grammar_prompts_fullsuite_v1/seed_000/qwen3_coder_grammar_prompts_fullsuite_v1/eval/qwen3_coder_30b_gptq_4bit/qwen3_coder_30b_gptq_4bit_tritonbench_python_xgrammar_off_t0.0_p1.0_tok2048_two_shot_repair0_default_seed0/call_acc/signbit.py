import torch
import triton
import triton.language as tl

@triton.jit
def _signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Extract sign bit using bit manipulation
    # For float32, we need to check the 31st bit (sign bit)
    # Convert to int32 to access individual bits
    x_int = tl.cast(x, tl.int32)
    # Extract sign bit (bit 31) and convert back to bool
    sign_bits = (x_int >> 31) & 1
    # Convert to boolean (True if sign bit is set)
    result = sign_bits == 1
    tl.store(out_ptr + offsets, result, mask=mask)

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    else:
        assert out.dtype == torch.bool, "Output tensor must have bool dtype"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _signbit_kernel[grid](input, out, n, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_signbit():
    results = {}

    # Test case 1: Positive and negative values
    input_tensor_1 = torch.tensor([1.0, -1.0, 0.0, -0.0], device='cuda')
    results["test_case_1"] = signbit(input_tensor_1)

    # Test case 2: All positive values
    input_tensor_2 = torch.tensor([3.5, 2.2, 0.1], device='cuda')
    results["test_case_2"] = signbit(input_tensor_2)

    # Test case 3: All negative values
    input_tensor_3 = torch.tensor([-3.5, -2.2, -0.1], device='cuda')
    results["test_case_3"] = signbit(input_tensor_3)

    # Test case 4: Mixed values with large numbers
    input_tensor_4 = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_4"] = signbit(input_tensor_4)

    return results

test_results = test_signbit()

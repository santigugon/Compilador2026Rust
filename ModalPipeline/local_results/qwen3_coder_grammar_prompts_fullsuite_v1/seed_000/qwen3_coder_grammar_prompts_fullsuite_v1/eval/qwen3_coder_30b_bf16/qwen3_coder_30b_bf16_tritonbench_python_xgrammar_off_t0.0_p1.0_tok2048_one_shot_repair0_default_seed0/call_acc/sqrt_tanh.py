import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.tanh(tl.sqrt(x))
    tl.store(y_ptr + offsets, y, mask=mask)

def sqrt_tanh(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == torch.float32, "Output tensor must be of type float32"
        assert out.device == input.device, "Output tensor must be on the same device as input tensor"
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

##################################################################################################################################################



import torch

def test_sqrt_tanh():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    results["test_case_1"] = sqrt_tanh(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-4.0, -9.0, -16.0], device='cuda')
    results["test_case_2"] = sqrt_tanh(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([4.0, -9.0, 16.0, -1.0], device='cuda')
    results["test_case_3"] = sqrt_tanh(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = sqrt_tanh(input4)

    return results

test_results = test_sqrt_tanh()

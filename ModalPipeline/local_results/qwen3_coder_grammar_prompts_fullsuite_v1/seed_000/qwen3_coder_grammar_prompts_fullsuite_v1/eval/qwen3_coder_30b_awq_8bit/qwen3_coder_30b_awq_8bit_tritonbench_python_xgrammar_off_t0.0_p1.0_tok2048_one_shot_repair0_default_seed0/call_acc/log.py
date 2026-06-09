import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.log(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def log(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != torch.float32:
            raise ValueError("Output tensor must have float32 dtype")
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    log_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out

##################################################################################################################################################



import torch

def test_log():
    results = {}

    # Test case 1: Basic test with positive values
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = log(input1)

    # Test case 2: Test with a tensor containing a zero
    input2 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_2"] = log(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = log(input3)

    # Test case 4: Test with a tensor containing a mix of positive, negative, and zero
    input4 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_4"] = log(input4)

    return results

test_results = test_log()

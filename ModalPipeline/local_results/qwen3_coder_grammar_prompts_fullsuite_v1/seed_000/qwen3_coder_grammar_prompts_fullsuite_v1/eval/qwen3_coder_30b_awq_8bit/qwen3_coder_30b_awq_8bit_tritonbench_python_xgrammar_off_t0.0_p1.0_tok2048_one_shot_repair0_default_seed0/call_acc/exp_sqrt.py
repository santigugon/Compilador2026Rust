import torch
import triton
import triton.language as tl

@triton.jit
def exp_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.sqrt(tl.exp(input))
    tl.store(output_ptr + offsets, output, mask=mask)

def exp_sqrt(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    assert input.dtype == torch.float32, "Input tensor must be float32"
    assert out.dtype == torch.float32, "Output tensor must be float32"
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    exp_sqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024
    )
    
    return out

##################################################################################################################################################



import torch

def test_exp_sqrt():
    results = {}

    # Test case 1: Basic functionality with a simple tensor
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_sqrt(input1)

    # Test case 2: Test with a tensor containing negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp_sqrt(input2)

    # Test case 3: Test with a tensor containing zero
    input3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp_sqrt(input3)

    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = exp_sqrt(input4, out=out4)

    return results

test_results = test_exp_sqrt()

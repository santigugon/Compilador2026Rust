import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_exp_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    sqrt_input = tl.sqrt(input)
    output = tl.exp(sqrt_input)
    tl.store(output_ptr + offsets, output, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    assert input.dtype == torch.float32, "Input tensor must be float32"
    assert out.dtype == torch.float32, "Output tensor must be float32"
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    sqrt_exp_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024
    )
    
    return out

##################################################################################################################################################



import torch

def test_sqrt_exp():
    results = {}

    # Test case 1: Basic functionality with GPU tensor
    a = torch.tensor([0.25, 1.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = sqrt_exp(a)

    # Test case 2: Empty tensor
    b = torch.tensor([], device='cuda')
    results["test_case_2"] = sqrt_exp(b)

    # Test case 3: Tensor with zero values
    c = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = sqrt_exp(c)

    # Test case 4: Using the out parameter
    d = torch.tensor([0.25, 1.0, 4.0, 9.0], device='cuda')
    out_tensor = torch.empty_like(d)
    results["test_case_4"] = sqrt_exp(d, out=out_tensor)

    return results

test_results = test_sqrt_exp()

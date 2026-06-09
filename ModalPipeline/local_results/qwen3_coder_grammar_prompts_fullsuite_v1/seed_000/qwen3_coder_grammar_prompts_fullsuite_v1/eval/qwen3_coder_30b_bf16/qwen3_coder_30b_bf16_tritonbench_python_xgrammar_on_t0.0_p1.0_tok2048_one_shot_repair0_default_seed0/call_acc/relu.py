import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, inplace: tl.constexpr):
    """Kernel for applying ReLU activation function."""
    pid = tl.program_id(0)
    block_start = pid * 1024
    block_end = min(block_start + 1024, n_elements)
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0.0)
    if inplace:
        tl.store(x_ptr + offsets, output, mask=mask)
    else:
        tl.store(output_ptr + offsets, output, mask=mask)


def relu(input, inplace=False):
    """Applies the rectified linear unit function element-wise."""
    output = torch.empty_like(input) if not inplace else input
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    relu_kernel[grid](input, output, n_elements, inplace)
    return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
#     return F.relu(input, inplace=inplace)

def test_relu():
    results = {}
    
    # Test case 1: Basic test with a simple tensor
    input1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = relu(input1)
    
    # Test case 2: Test with a 2D tensor
    input2 = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device='cuda')
    results["test_case_2"] = relu(input2)
    
    # Test case 3: Test with inplace=True
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = relu(input3, inplace=True)
    
    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], device='cuda')
    results["test_case_4"] = relu(input4)
    
    return results

test_results = test_relu()

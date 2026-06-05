import torch
import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    inplace: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input * other
    output = tl.maximum(output, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def mul_relu(input, other, inplace=False, out=None) -> torch.Tensor:
    if inplace and out is not None:
        raise ValueError("Cannot specify both 'inplace' and 'out'")
    
    if inplace and not isinstance(other, torch.Tensor):
        raise ValueError("Inplace operation requires 'other' to be a tensor")
    
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    if out is not None:
        output = out
    elif inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    if input.dtype != other.dtype:
        other = other.to(input.dtype)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    if inplace:
        mul_relu_kernel[grid](
            input,
            other,
            input,
            n_elements,
            BLOCK_SIZE,
            True,
        )
    else:
        mul_relu_kernel[grid](
            input,
            other,
            output,
            n_elements,
            BLOCK_SIZE,
            False,
        )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def mul_relu(input, other, inplace=False, out=None):
#     result = torch.mul(input, other)
#     return F.relu(result, inplace=inplace)

def test_mul_relu():
    results = {}

    # Test case 1: Basic multiplication and ReLU with two tensors
    input1 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other1 = torch.tensor([1.0, -1.0, 1.0, -1.0], device='cuda')
    results["test_case_1"] = mul_relu(input1, other1)

    # Test case 2: Multiplication with a scalar
    input2 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = mul_relu(input2, other2)

    # Test case 3: In-place operation
    input3 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other3 = torch.tensor([1.0, -1.0, 1.0, -1.0], device='cuda')
    results["test_case_3"] = mul_relu(input3, other3, inplace=True)

    # Test case 4: Multiplication with a different shaped tensor
    input4 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[1.0, -1.0], [1.0, -1.0]], device='cuda')
    results["test_case_4"] = mul_relu(input4, other4)

    return results

test_results = test_mul_relu()

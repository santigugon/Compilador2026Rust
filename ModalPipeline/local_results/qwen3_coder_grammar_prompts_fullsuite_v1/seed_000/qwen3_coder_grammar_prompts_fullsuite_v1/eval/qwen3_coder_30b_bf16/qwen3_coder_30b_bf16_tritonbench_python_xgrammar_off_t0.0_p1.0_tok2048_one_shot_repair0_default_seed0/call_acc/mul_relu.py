import torch
import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size,
    inplace: tl.constexpr,
    other_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_offsets = offset + tl.arange(0, BLOCK_SIZE)
    other_offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    input_mask = input_offsets < input_size
    
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask)
    
    if other_is_scalar:
        other_vals = tl.full((BLOCK_SIZE,), other_ptr[0], dtype=tl.float32)
    else:
        other_vals = tl.load(other_ptr + other_offsets, mask=input_mask)
    
    mul_result = input_vals * other_vals
    relu_result = tl.maximum(mul_result, 0.0)
    
    if inplace:
        tl.store(input_ptr + input_offsets, relu_result, mask=input_mask)
    else:
        tl.store(output_ptr + input_offsets, relu_result, mask=input_mask)

def mul_relu(input, other, inplace=False, out=None):
    if inplace and out is not None:
        raise ValueError("Cannot specify both 'inplace=True' and 'out' parameter")
    
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if not isinstance(other, (torch.Tensor, int, float)):
        raise TypeError("other must be a torch.Tensor, int, or float")
    
    if input.dtype != torch.float32:
        raise TypeError("input must be of type torch.float32")
    
    if not isinstance(other, torch.Tensor) and not isinstance(other, (int, float)):
        raise TypeError("other must be a torch.Tensor, int, or float")
    
    if isinstance(other, torch.Tensor) and other.dtype != torch.float32:
        raise TypeError("other must be of type torch.float32")
    
    if input.device != torch.device('cuda'):
        raise ValueError("input must be on CUDA device")
    
    if isinstance(other, torch.Tensor) and other.device != torch.device('cuda'):
        raise ValueError("other must be on CUDA device")
    
    if inplace and not input.requires_grad:
        output = input
    elif out is not None:
        output = out
        if output.device != torch.device('cuda'):
            raise ValueError("output must be on CUDA device")
        if output.dtype != torch.float32:
            raise TypeError("output must be of type torch.float32")
    else:
        output = torch.empty_like(input, device='cuda', dtype=torch.float32)
    
    if input.numel() == 0:
        return output
    
    BLOCK_SIZE = 1024
    grid_size = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    other_is_scalar = not isinstance(other, torch.Tensor)
    other_ptr = other if other_is_scalar else other.data_ptr()
    
    mul_relu_kernel[grid_size](
        input.data_ptr(),
        other_ptr,
        output.data_ptr(),
        input.numel(),
        inplace,
        other_is_scalar,
        BLOCK_SIZE
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

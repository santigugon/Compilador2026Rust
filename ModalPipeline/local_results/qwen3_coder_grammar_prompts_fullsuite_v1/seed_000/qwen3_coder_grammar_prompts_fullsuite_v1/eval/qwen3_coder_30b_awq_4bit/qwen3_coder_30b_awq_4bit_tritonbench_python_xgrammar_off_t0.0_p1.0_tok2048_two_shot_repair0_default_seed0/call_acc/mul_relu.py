import torch
import triton
import triton.language as tl

@triton.jit
def _mul_relu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, inplace: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x * y
    # Apply ReLU
    relu_result = tl.maximum(result, 0.0)
    tl.store(out_ptr + offsets, relu_result, mask=mask)

def mul_relu(input, other, inplace=False, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure tensors are on the same device and have compatible shapes
    if other.device != input.device:
        other = other.to(input.device)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        pass  # The kernel will handle this with proper masking
    
    # Determine output tensor
    if out is not None:
        output = out
    elif inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Compute output shape
    if input.shape != other.shape:
        # Use torch's broadcasting to get the result shape
        output_shape = torch.broadcast_tensors(input, other)[0].shape
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle inplace case
    if inplace and out is None:
        # For inplace, we need to make sure we're not modifying the wrong tensor
        # We'll use a temporary tensor for the computation
        temp_output = torch.empty_like(input)
        _mul_relu_kernel[grid](input, other, temp_output, n, BLOCK=block, inplace=inplace)
        output.copy_(temp_output)
        return output
    else:
        # For non-inplace or when out is specified
        if out is not None:
            _mul_relu_kernel[grid](input, other, out, n, BLOCK=block, inplace=inplace)
            return out
        else:
            _mul_relu_kernel[grid](input, other, output, n, BLOCK=block, inplace=inplace)
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

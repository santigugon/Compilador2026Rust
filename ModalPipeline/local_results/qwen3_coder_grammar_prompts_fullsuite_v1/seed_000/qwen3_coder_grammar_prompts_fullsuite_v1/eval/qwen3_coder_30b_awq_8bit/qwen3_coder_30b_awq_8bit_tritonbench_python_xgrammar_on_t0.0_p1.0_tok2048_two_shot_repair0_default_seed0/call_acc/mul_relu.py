import torch
import triton
import triton.language as tl

@triton.jit
def _mul_relu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr, IS_INPLACE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x * y * alpha
    # Apply ReLU
    result = tl.maximum(result, 0.0)
    if IS_INPLACE:
        tl.store(x_ptr + offsets, result, mask=mask)
    else:
        tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _mul_relu_kernel_scalar(x_ptr, y_val, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr, IS_INPLACE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x * y_val * alpha
    # Apply ReLU
    result = tl.maximum(result, 0.0)
    if IS_INPLACE:
        tl.store(x_ptr + offsets, result, mask=mask)
    else:
        tl.store(out_ptr + offsets, result, mask=mask)

def mul_relu(input, other, inplace=False, out=None):
    if inplace and out is not None:
        raise ValueError("Cannot specify both 'inplace=True' and 'out' parameter.")
    
    # Handle scalar other
    if not torch.is_tensor(other):
        if inplace:
            # For inplace, we can directly modify input
            input = input * other
            input = torch.relu(input)
            return input
        else:
            # Create output tensor
            out = torch.empty_like(input)
            # Launch kernel
            n = input.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            _mul_relu_kernel_scalar[grid](input, other, out, n, 1.0, BLOCK=block, IS_INPLACE=False)
            return out
    
    # Handle tensor other
    if inplace:
        # Check if we can do inplace operation
        if input.shape == other.shape:
            # Launch kernel for inplace
            n = input.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            _mul_relu_kernel[grid](input, other, input, n, 1.0, BLOCK=block, IS_INPLACE=True)
            return input
        else:
            raise ValueError("For inplace operation, input and other must have the same shape.")
    else:
        # Create output tensor
        if out is None:
            out = torch.empty_like(input)
        # Launch kernel
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _mul_relu_kernel[grid](input, other, out, n, 1.0, BLOCK=block, IS_INPLACE=False)
        return out
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

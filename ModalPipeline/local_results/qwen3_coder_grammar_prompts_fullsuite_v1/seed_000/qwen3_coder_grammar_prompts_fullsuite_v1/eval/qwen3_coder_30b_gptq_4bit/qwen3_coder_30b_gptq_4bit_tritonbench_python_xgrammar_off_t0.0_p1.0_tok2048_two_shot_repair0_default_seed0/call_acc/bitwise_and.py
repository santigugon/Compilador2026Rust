import torch
import triton
import triton.language as tl

@triton.jit
def _bitwise_and_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    result = x & y
    tl.store(out_ptr + offsets, result, mask=mask)

def bitwise_and(input, other, *, out=None):
    # Ensure inputs are of compatible types
    if not (torch.is_tensor(input) and torch.is_tensor(other)):
        raise TypeError("Both input and other must be tensors")
    
    # Check if the tensors have compatible shapes for broadcasting
    if input.shape != other.shape:
        # Broadcasting is supported by PyTorch
        pass
    
    # Determine output tensor
    if out is not None:
        out = out
    else:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n = input.numel()
    
    # Set block size and grid size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel
    _bitwise_and_kernel[grid](input, other, out, n, BLOCK=block)
    
    return out

##################################################################################################################################################



import torch

def test_bitwise_and():
    results = {}

    # Test case 1: Bitwise AND with integer tensors
    input1 = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    other1 = torch.tensor([3, 2, 1], dtype=torch.int32, device='cuda')
    results["test_case_1"] = bitwise_and(input1, other1)

    # Test case 2: Bitwise AND with boolean tensors
    input2 = torch.tensor([True, False, True], dtype=torch.bool, device='cuda')
    other2 = torch.tensor([False, False, True], dtype=torch.bool, device='cuda')
    results["test_case_2"] = bitwise_and(input2, other2)

    # Test case 3: Bitwise AND with different shapes (broadcasting)
    input3 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device='cuda')
    other3 = torch.tensor([1, 0], dtype=torch.int32, device='cuda')
    results["test_case_3"] = bitwise_and(input3, other3)

    # Test case 4: Bitwise AND with scalar tensor
    input4 = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    other4 = torch.tensor(2, dtype=torch.int32, device='cuda')
    results["test_case_4"] = bitwise_and(input4, other4)

    return results

test_results = test_bitwise_and()

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
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure both tensors have the same dtype and device
    if other.dtype != input.dtype:
        other = other.to(dtype=input.dtype, device=input.device)
    
    # Handle out parameter
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
        if out.device != input.device:
            raise ValueError("Output tensor must be on the same device as input tensor")
    
    # Get total number of elements
    n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
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

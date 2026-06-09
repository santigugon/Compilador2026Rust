import torch
import triton
import triton.language as tl

@triton.jit
def _mul_sub_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    result = x * y - alpha * z
    tl.store(out_ptr + offsets, result, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar inputs
    if not torch.is_tensor(other_mul):
        other_mul = torch.tensor(other_mul, dtype=input.dtype, device=input.device)
    if not torch.is_tensor(other_sub):
        other_sub = torch.tensor(other_sub, dtype=input.dtype, device=input.device)
    
    # Ensure tensors are contiguous for kernel access
    input = input.contiguous()
    other_mul = other_mul.contiguous()
    other_sub = other_sub.contiguous()
    out = out.contiguous()
    
    _mul_sub_kernel[grid](input, other_mul, other_sub, out, n, alpha, BLOCK=block)
    return out
##################################################################################################################################################



import torch

def test_mul_sub():
    results = {}

    # Test case 1: input, other_mul, other_sub are tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_mul_tensor = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    other_sub_tensor = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = mul_sub(input_tensor, other_mul_tensor, other_sub_tensor)

    # Test case 2: input is a tensor, other_mul is a number, other_sub is a tensor
    other_mul_number = 2.0
    results["test_case_2"] = mul_sub(input_tensor, other_mul_number, other_sub_tensor)

    # Test case 3: input is a tensor, other_mul is a tensor, other_sub is a number
    other_sub_number = 0.5
    results["test_case_3"] = mul_sub(input_tensor, other_mul_tensor, other_sub_number)

    # Test case 4: input, other_mul, other_sub are numbers
    input_number = 3.0
    results["test_case_4"] = mul_sub(input_number, other_mul_number, other_sub_number)

    return results

test_results = test_mul_sub()

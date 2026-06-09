import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Extract sign bit: cast to int and check bit 63 (for float64) or 31 (for float32)
    # For simplicity, we'll use a method that works for both
    x_bits = tl.bitcast(x, tl.int32)
    sign_bit = (x_bits >> 31) & 1
    tl.store(out_ptr + offsets, sign_bit.to(tl.int1), mask=mask)

@triton.jit
def _bitwise_and_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    result = x & y
    tl.store(out_ptr + offsets, result, mask=mask)


def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Check if input is floating point
    if input.dtype.is_floating_point:
        # For signbit, we need to check the sign bit
        signbit_result = torch.empty_like(input, dtype=torch.bool)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _signbit_kernel[grid](input, signbit_result, n, BLOCK=block)
    else:
        # If not floating point, signbit is not applicable
        signbit_result = torch.zeros(input.shape, dtype=torch.bool, device=input.device)
    
    # For bitwise_and, we need both tensors to be integral or boolean
    if not (other.dtype.is_integral or other.dtype == torch.bool):
        raise ValueError("other tensor must be of integral or boolean type for bitwise_and")
    
    bitwise_and_result = torch.empty_like(other)
    n = other.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _bitwise_and_kernel[grid](input, other, bitwise_and_result, n, BLOCK=block)
    
    return signbit_result, bitwise_and_result
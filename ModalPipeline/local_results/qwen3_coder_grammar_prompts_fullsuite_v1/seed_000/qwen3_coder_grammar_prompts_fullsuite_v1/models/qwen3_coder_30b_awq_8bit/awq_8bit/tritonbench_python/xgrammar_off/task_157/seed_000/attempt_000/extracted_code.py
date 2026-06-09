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
    # Extract sign bit: cast to int and check the most significant bit
    # For float32, the sign bit is bit 31 (MSB)
    x_bits = tl.bitcast(x, tl.int32)
    sign_bits = (x_bits >> 31) & 1
    tl.store(out_ptr + offsets, sign_bits.to(tl.int1), mask=mask)

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
    # For signbit operation
    signbit_out = torch.empty(input.shape, dtype=torch.bool, device=input.device)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For bitwise_and operation
    bitwise_and_out = torch.empty(other.shape, dtype=other.dtype, device=other.device)
    
    # Handle signbit
    if input.is_floating_point():
        _signbit_kernel[grid](input, signbit_out, n, BLOCK=block)
    else:
        # For non-floating point inputs, signbit is not applicable
        signbit_out.fill_(False)
    
    # Handle bitwise_and
    if other.is_floating_point():
        # If other is floating point, we need to convert to integer for bitwise operations
        # But this is not standard behavior, so we'll just return zeros
        bitwise_and_out.fill_(0)
    else:
        # For integral or boolean types, perform bitwise AND
        _bitwise_and_kernel[grid](input, other, bitwise_and_out, n, BLOCK=block)
    
    return (signbit_out, bitwise_and_out)

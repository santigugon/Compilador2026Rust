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
    # Extract sign bit: cast to int and check the sign bit
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
    # Ensure input is float32 or float64
    if input.dtype not in [torch.float32, torch.float64]:
        raise ValueError("Input tensor must be of floating point type")
    
    # Ensure other is integral or boolean type
    if not (other.dtype.is_integral or other.dtype == torch.bool):
        raise ValueError("Other tensor must be of integral or boolean type")
    
    out1 = torch.empty(input.shape, dtype=torch.bool, device=input.device)
    out2 = torch.empty(other.shape, dtype=other.dtype, device=other.device)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Compute signbit
    _signbit_kernel[grid](input, out1, n, BLOCK=block)
    
    # Compute bitwise_and
    _bitwise_and_kernel[grid](input.to(other.dtype), other, out2, n, BLOCK=block)
    
    return (out1, out2)
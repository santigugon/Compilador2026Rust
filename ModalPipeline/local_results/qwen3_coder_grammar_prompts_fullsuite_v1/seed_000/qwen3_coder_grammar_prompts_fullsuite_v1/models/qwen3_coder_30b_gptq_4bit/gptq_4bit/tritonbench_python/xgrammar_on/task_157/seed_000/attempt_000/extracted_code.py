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
    # Extract sign bit using bit manipulation
    # For float32, sign bit is bit 31
    # Convert to int32 to access bits
    x_int = tl.cast(x, tl.int32)
    # Extract sign bit (bit 31)
    sign_bits = (x_int >> 31) & 1
    # Convert back to boolean
    y = sign_bits == 1
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _bitwise_and_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    # Perform bitwise AND
    result = x & y
    tl.store(out_ptr + offsets, result, mask=mask)


def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate input types
    if input.dtype in [torch.float32, torch.float64]:
        # For floating point, we need to check sign bit
        pass
    else:
        raise ValueError("First input must be of floating point type")
    
    if not (other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]):
        raise ValueError("Second input must be of integral or boolean type")
    
    # Ensure tensors have the same shape
    assert input.shape == other.shape, "Input tensors must have the same shape"
    
    # Create output tensors
    signbit_result = torch.empty_like(input, dtype=torch.bool)
    bitwise_and_result = torch.empty_like(other)
    
    # Get total number of elements
    n = input.numel()
    
    # Set block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernels
    _signbit_kernel[grid](input, signbit_result, n, BLOCK=block)
    _bitwise_and_kernel[grid](input, other, bitwise_and_result, n, BLOCK=block)
    
    return (signbit_result, bitwise_and_result)
import torch
import triton
import triton.language as tl

@triton.jit
def _signbit_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # For floating point, check the sign bit by casting to integer and checking bit 31 (or 63 for double)
    # We use bit manipulation to extract sign bit
    x_int = tl.cast(x, tl.int32)
    sign_bit = (x_int >> 31) & 1
    tl.store(out_ptr + offsets, sign_bit, mask=mask)

@triton.jit
def _bitwise_and_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    result = x & y
    tl.store(out_ptr + offsets, result, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure tensors are contiguous for Triton
    input = input.contiguous()
    other = other.contiguous()
    
    # Create output tensors
    signbit_result = torch.empty_like(input, dtype=torch.bool)
    bitwise_and_result = torch.empty_like(other, dtype=other.dtype)
    
    # Get total number of elements
    n = input.numel()
    
    # Determine block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernels
    _signbit_kernel[grid](input, signbit_result, n, BLOCK=block)
    _bitwise_and_kernel[grid](input, other, bitwise_and_result, n, BLOCK=block)
    
    return signbit_result, bitwise_and_result

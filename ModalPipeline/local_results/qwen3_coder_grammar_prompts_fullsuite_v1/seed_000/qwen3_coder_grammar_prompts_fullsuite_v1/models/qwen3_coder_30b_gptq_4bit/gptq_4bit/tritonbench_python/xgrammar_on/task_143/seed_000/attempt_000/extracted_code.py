import torch
import triton
import triton.language as tl

def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    # Handle scalar inputs
    if input.dim() == 0:
        input = input.unsqueeze(0)
        out = out.unsqueeze(0)
    
    # Get the number of elements
    n = input.numel()
    
    # Define block size
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    
    # Get the dtype of the input tensor
    dtype = input.dtype
    
    # For different dtypes, we need to handle the sign bit differently
    if dtype == torch.float32:
        _signbit_kernel_f32[grid](input, out, n, BLOCK=BLOCK)
    elif dtype == torch.float64:
        _signbit_kernel_f64[grid](input, out, n, BLOCK=BLOCK)
    else:
        # For other dtypes, fall back to PyTorch implementation
        out = torch.signbit(input)
    
    # Return the output tensor
    if input.dim() == 0:
        return out.squeeze(0)
    return out

@triton.jit
def _signbit_kernel_f32(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # For float32, the sign bit is the most significant bit of the 32-bit representation
    # Convert to int32 to access the bits
    x_int = tl.cast(x, tl.int32)
    # Extract the sign bit (bit 31)
    sign_bit = (x_int >> 31) & 1
    # Store the result
    tl.store(out_ptr + offsets, sign_bit, mask=mask)

@triton.jit
def _signbit_kernel_f64(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # For float64, the sign bit is the most significant bit of the 64-bit representation
    # Convert to int64 to access the bits
    x_int = tl.cast(x, tl.int64)
    # Extract the sign bit (bit 63)
    sign_bit = (x_int >> 63) & 1
    # Store the result
    tl.store(out_ptr + offsets, sign_bit, mask=mask)
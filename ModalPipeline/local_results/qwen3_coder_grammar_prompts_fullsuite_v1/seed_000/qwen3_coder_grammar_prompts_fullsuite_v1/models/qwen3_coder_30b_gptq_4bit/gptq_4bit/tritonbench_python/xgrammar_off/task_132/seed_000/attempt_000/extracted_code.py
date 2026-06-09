import torch
import triton
import triton.language as tl

@triton.jit
def _mul_sub_kernel(x_ptr, y_mul_ptr, y_sub_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_mul = tl.load(y_mul_ptr + offsets, mask=mask, other=0.0)
    y_sub = tl.load(y_sub_ptr + offsets, mask=mask, other=0.0)
    result = x * y_mul - alpha * y_sub
    tl.store(out_ptr + offsets, result, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar inputs
    if not torch.is_tensor(other_mul):
        other_mul = torch.tensor(other_mul, dtype=input.dtype, device=input.device)
    if not torch.is_tensor(other_sub):
        other_sub = torch.tensor(other_sub, dtype=input.dtype, device=input.device)
    
    # Ensure tensors have the same device and dtype
    other_mul = other_mul.to(input.device).to(input.dtype)
    other_sub = other_sub.to(input.device).to(input.dtype)
    
    # Expand dimensions if needed for broadcasting
    if other_mul.dim() == 0:
        other_mul = other_mul.expand_as(input)
    if other_sub.dim() == 0:
        other_sub = other_sub.expand_as(input)
    
    # Handle broadcasting
    if other_mul.shape != input.shape:
        other_mul = other_mul.expand_as(input)
    if other_sub.shape != input.shape:
        other_sub = other_sub.expand_as(input)
    
    _mul_sub_kernel[grid](input, other_mul, other_sub, out, n, alpha, BLOCK=block)
    return out

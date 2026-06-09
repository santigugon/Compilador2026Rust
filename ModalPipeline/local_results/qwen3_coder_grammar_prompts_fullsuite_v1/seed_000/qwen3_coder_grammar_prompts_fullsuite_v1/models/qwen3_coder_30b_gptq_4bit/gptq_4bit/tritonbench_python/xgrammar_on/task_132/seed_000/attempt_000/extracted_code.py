import torch
import triton
import triton.language as tl

@triton.jit
def _mul_sub_kernel(x_ptr, y_mul_ptr, y_sub_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr, y_mul_is_scalar: tl.constexpr, y_sub_is_scalar: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if y_mul_is_scalar:
        y_mul = alpha  # In this case, alpha is used as the scalar value
    else:
        y_mul = tl.load(y_mul_ptr + offsets, mask=mask, other=0.0)
    
    if y_sub_is_scalar:
        y_sub = alpha  # In this case, alpha is used as the scalar value
    else:
        y_sub = tl.load(y_sub_ptr + offsets, mask=mask, other=0.0)
    
    result = x * y_mul - y_sub
    tl.store(out_ptr + offsets, result, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine if other_mul and other_sub are scalars
    y_mul_is_scalar = not torch.is_tensor(other_mul)
    y_sub_is_scalar = not torch.is_tensor(other_sub)
    
    # Handle scalar cases
    if y_mul_is_scalar:
        other_mul = torch.tensor(other_mul, dtype=input.dtype, device=input.device)
    if y_sub_is_scalar:
        other_sub = torch.tensor(other_sub, dtype=input.dtype, device=input.device)
    
    # Ensure tensors are contiguous
    if not other_mul.is_contiguous():
        other_mul = other_mul.contiguous()
    if not other_sub.is_contiguous():
        other_sub = other_sub.contiguous()
    
    # If other_mul or other_sub is a scalar, we need to broadcast it
    if y_mul_is_scalar:
        other_mul = other_mul.expand_as(input)
    if y_sub_is_scalar:
        other_sub = other_sub.expand_as(input)
    
    _mul_sub_kernel[grid](input, other_mul, other_sub, out, n, alpha, BLOCK=block, y_mul_is_scalar=y_mul_is_scalar, y_sub_is_scalar=y_sub_is_scalar)
    return out
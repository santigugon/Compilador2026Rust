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

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
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
    
    # Ensure tensors have the same device and dtype
    other_mul = other_mul.to(input.device, input.dtype)
    other_sub = other_sub.to(input.device, input.dtype)
    
    # Handle broadcasting
    if other_mul.numel() == 1:
        other_mul = other_mul.expand(input.shape)
    if other_sub.numel() == 1:
        other_sub = other_sub.expand(input.shape)
    
    _mul_sub_kernel[grid](
        input, 
        other_mul, 
        other_sub, 
        out, 
        n, 
        alpha, 
        BLOCK=block
    )
    
    return out

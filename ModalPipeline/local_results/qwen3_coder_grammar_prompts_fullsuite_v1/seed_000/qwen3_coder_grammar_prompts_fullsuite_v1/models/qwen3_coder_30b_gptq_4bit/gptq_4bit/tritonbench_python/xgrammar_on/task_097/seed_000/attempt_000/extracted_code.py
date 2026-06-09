import torch
import triton
import triton.language as tl

def bitwise_and(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        if input.dtype == torch.bool:
            return input & other
        else:
            return input & other
    
    # Ensure tensors have the same dtype
    if input.dtype != other.dtype:
        raise ValueError("Input and other must have the same dtype")
    
    # For boolean tensors, use logical AND
    if input.dtype == torch.bool:
        # Use torch's native implementation for boolean tensors
        return torch.bitwise_and(input, other, out=out)
    
    # For integral types, use Triton kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    @triton.jit
    def _bitwise_and_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        result = x & y
        tl.store(out_ptr + offsets, result, mask=mask)
    
    _bitwise_and_kernel[grid](input, other, out, n, BLOCK=block)
    return out
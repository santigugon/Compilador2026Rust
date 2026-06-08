import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + alpha * y, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        if out is not None:
            return torch.add(input, other, alpha=alpha, out=out)
        else:
            return torch.add(input, other, alpha=alpha)
    
    # Ensure tensors have compatible shapes for broadcasting
    if input.shape == other.shape:
        # Same shape, direct element-wise addition
        out_tensor = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _add_kernel[grid](input, other, out_tensor, n, alpha, BLOCK=block)
        if out is not None:
            out.copy_(out_tensor)
            return out
        return out_tensor
    else:
        # Broadcasting case - use PyTorch's native implementation
        if out is not None:
            return torch.add(input, other, alpha=alpha, out=out)
        else:
            return torch.add(input, other, alpha=alpha)

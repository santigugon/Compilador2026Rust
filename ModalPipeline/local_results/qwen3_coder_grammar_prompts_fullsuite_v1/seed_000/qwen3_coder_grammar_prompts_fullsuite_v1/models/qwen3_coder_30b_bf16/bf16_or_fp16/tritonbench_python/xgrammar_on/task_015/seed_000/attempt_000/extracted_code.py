import torch
import triton
import triton.language as tl

def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr, is_scalar: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if is_scalar:
        y = alpha * y_ptr[0]
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            return torch.add(input, other, alpha=alpha, out=out)
        else:
            return torch.add(input, other, alpha=alpha)
    
    # For tensor other, ensure same device and dtype
    if input.device != other.device:
        other = other.to(input.device)
    
    # Handle broadcasting
    try:
        # Use torch's broadcasting to get the result shape
        result_shape = torch.broadcast_shapes(input.shape, other.shape)
    except RuntimeError:
        raise ValueError("Input shapes are not broadcastable")
    
    # Create output tensor
    if out is not None:
        if out.shape != result_shape or out.dtype != input.dtype:
            raise ValueError("Output tensor has incorrect shape or dtype")
        out = out
    else:
        out = torch.empty(result_shape, dtype=input.dtype, device=input.device)
    
    # If shapes match exactly, use direct kernel
    if input.shape == other.shape:
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _add_kernel[grid](input, other, out, n, alpha, BLOCK=block, is_scalar=False)
    else:
        # For broadcasting, use torch operations
        out = torch.add(input, other, alpha=alpha, out=out)
    
    return out
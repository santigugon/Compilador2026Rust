import torch
import triton
import triton.language as tl

def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr, x_is_scalar: tl.constexpr, y_is_scalar: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    if x_is_scalar:
        x = tl.load(x_ptr)
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if y_is_scalar:
        y = tl.load(y_ptr)
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle broadcasting
    if input.shape == other.shape:
        # No broadcasting needed
        broadcast_shape = input.shape
    else:
        # Use torch's broadcasting rules
        broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    
    # Create output tensor
    if out is not None:
        out = out.clone()  # Ensure it's a new tensor
    else:
        out = torch.empty(broadcast_shape, dtype=input.dtype, device=input.device)
    
    # Flatten tensors for kernel execution
    input_flat = input.reshape(-1)
    other_flat = other.reshape(-1)
    out_flat = out.reshape(-1)
    
    n = out_flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine if inputs are scalars
    x_is_scalar = input.numel() == 1
    y_is_scalar = other.numel() == 1
    
    _add_kernel[grid](input_flat, other_flat, out_flat, n, alpha, BLOCK=block, x_is_scalar=x_is_scalar, y_is_scalar=y_is_scalar)
    
    return out
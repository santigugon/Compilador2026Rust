import torch
import triton
import triton.language as tl

def add(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            return torch.add(input, other, alpha=alpha, out=out)
        else:
            return torch.add(input, other, alpha=alpha)
    
    # Ensure input and other have compatible shapes for broadcasting
    # For simplicity, we'll use PyTorch's native implementation for broadcasting
    # and focus on the core Triton kernel for element-wise operations
    if out is not None:
        # If out is provided, we need to handle it carefully
        # For now, we'll delegate to PyTorch for complex cases
        return torch.add(input, other, alpha=alpha, out=out)
    
    # Get the output tensor with correct shape and dtype
    out = torch.empty_like(input)
    
    # Flatten tensors for processing
    input_flat = input.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    # Determine the total number of elements
    n = input_flat.numel()
    
    # If other is a scalar, we can handle it directly
    if other_flat.numel() == 1:
        # Use a simple kernel for scalar addition
        _add_scalar_kernel[(triton.cdiv(n, 256),)](input_flat, other_flat, out_flat, n, alpha)
    else:
        # For tensor-tensor addition, use the standard kernel
        _add_kernel[(triton.cdiv(n, 256),)](input_flat, other_flat, out_flat, n, alpha)
    
    return out

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr = 256):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _add_scalar_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr = 256):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)
import torch
import triton
import triton.language as tl

def _broadcast_shapes(shape1, shape2):
    # Helper to compute broadcasted shape
    # This is a simplified version for the purpose of this kernel
    # In practice, you'd want to use PyTorch's broadcasting rules
    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    elif len(shape2) > len(shape1):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    
    broadcasted = []
    for s1, s2 in zip(shape1, shape2):
        if s1 == 1:
            broadcasted.append(s2)
        elif s2 == 1:
            broadcasted.append(s1)
        else:
            if s1 != s2:
                raise ValueError("Shapes are not broadcastable")
            broadcasted.append(s1)
    return broadcasted

@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x * y
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _mul_kernel_scalar(x_ptr, scalar, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x * scalar
    tl.store(out_ptr + offsets, result, mask=mask)

def mul(input, other, *, out=None):
    # Handle scalar case
    if not torch.is_tensor(other):
        if out is not None:
            # If out is provided, we need to compute the result and store it
            result = input * other
            out.copy_(result)
            return out
        else:
            return input * other
    
    # Handle tensor case
    # Determine output shape
    if input.shape == other.shape:
        # Direct element-wise multiplication
        output_shape = input.shape
    else:
        # Use PyTorch's broadcasting rules
        try:
            # This is a simplified approach - in practice, you'd want to use
            # PyTorch's broadcasting logic or implement a more robust version
            output_shape = torch.broadcast_shapes(input.shape, other.shape)
        except Exception:
            raise ValueError("Shapes are not broadcastable")
    
    # Create output tensor
    if out is not None:
        if out.shape != output_shape:
            raise ValueError("Output tensor shape does not match expected broadcasted shape")
        out_tensor = out
    else:
        out_tensor = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # If shapes are the same and we can do direct element-wise
    if input.shape == other.shape:
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _mul_kernel[grid](input, other, out_tensor, n, BLOCK=block)
    else:
        # For broadcasting, we need to handle it differently
        # This is a simplified approach - in practice, you'd want to implement
        # proper broadcasting logic in the kernel
        out_tensor = input * other
        
    return out_tensor
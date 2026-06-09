import torch
import triton
import triton.language as tl

def _broadcast_shapes(shape1, shape2):
    # Helper to compute broadcasted shape
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
    shape2 = [1] * (len(shape1) - len(shape2)) + list(shape2)
    result = []
    for s1, s2 in zip(shape1, shape2):
        if s1 == 1:
            result.append(s2)
        elif s2 == 1:
            result.append(s1)
        else:
            if s1 != s2:
                raise ValueError("Shapes are not broadcastable")
            result.append(s1)
    return tuple(result)

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
def _mul_scalar_kernel(x_ptr, scalar: tl.constexpr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
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
            # For scalar multiplication with out tensor, we need to handle it carefully
            # This is a simplified approach - in practice, we'd want to avoid this path
            # if possible, but for compatibility we'll do it the PyTorch way
            return torch.mul(input, other, out=out)
        else:
            return input * other
    
    # Handle tensor case
    if out is None:
        # Compute output shape
        input_shape = input.shape
        other_shape = other.shape
        try:
            broadcast_shape = _broadcast_shapes(input_shape, other_shape)
        except ValueError:
            raise ValueError("Input shapes are not broadcastable")
        
        # Create output tensor with correct shape
        out = torch.empty(broadcast_shape, dtype=torch.result_type(input, other), device=input.device)
    else:
        # Validate that out tensor has correct shape
        if input.shape != other.shape:
            try:
                broadcast_shape = _broadcast_shapes(input.shape, other.shape)
                if out.shape != broadcast_shape:
                    raise ValueError("Output tensor shape does not match broadcasted input shapes")
            except ValueError:
                raise ValueError("Input shapes are not broadcastable")
        
    # Get total number of elements
    n = out.numel()
    if n == 0:
        return out
    
    # Determine block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel
    _mul_kernel[grid](input, other, out, n, BLOCK=block)
    return out
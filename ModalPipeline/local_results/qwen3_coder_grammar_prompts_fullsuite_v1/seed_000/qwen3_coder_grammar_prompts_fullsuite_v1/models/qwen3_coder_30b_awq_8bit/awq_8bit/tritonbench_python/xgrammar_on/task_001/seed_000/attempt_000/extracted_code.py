import torch
import triton
import triton.language as tl

def _get_rounding_mode(mode):
    if mode is None:
        return 0
    elif mode == 'trunc':
        return 1
    elif mode == 'floor':
        return 2
    else:
        raise ValueError(f"Unsupported rounding_mode: {mode}")

@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division
    result = x / y
    
    # Apply rounding if specified
    if rounding_mode == 1:  # trunc
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 2:  # floor
        result = tl.floor(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _div_kernel_scalar(x_ptr, y_val, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division
    result = x / y_val
    
    # Apply rounding if specified
    if rounding_mode == 1:  # trunc
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 2:  # floor
        result = tl.floor(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            # For scalar division with output tensor, we need to handle it carefully
            # Create a tensor with the same shape as input for the scalar
            other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
            return div(input, other_tensor, rounding_mode=rounding_mode, out=out)
        else:
            # Simple scalar division
            if rounding_mode is None:
                return input / other
            else:
                # For scalar division with rounding, we need to use a different approach
                # This is a simplified version that handles the most common case
                result = input / other
                if rounding_mode == 'trunc':
                    return torch.trunc(result)
                elif rounding_mode == 'floor':
                    return torch.floor(result)
                else:
                    return result
    
    # Handle tensor division
    if out is not None:
        # If output tensor is provided, we need to ensure it's compatible
        # For now, we'll just compute the result and copy to output
        result = div(input, other, rounding_mode=rounding_mode)
        out.copy_(result)
        return out
    
    # Get the rounding mode
    rounding_mode_int = _get_rounding_mode(rounding_mode)
    
    # Ensure tensors are compatible for element-wise operations
    input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Get total number of elements
    n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if rounding_mode is None:
        # Simple division without rounding
        _div_kernel[grid](input, other, out, n, 0, BLOCK=block)
    else:
        # Division with rounding
        _div_kernel[grid](input, other, out, n, rounding_mode_int, BLOCK=block)
    
    return out
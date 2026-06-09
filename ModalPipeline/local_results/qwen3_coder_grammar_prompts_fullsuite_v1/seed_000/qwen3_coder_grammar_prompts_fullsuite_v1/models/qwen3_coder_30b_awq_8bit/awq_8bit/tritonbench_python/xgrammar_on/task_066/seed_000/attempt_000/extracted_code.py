import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    gelu = x * 0.5 * (1.0 + tl.tanh(tanh_arg))
    tl.store(out_ptr + offsets, gelu, mask=mask)

def _masked_select_add_gelu_kernel(x_ptr, mask_ptr, other_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=False)
    # Select elements based on mask
    selected_x = tl.where(mask_val, x, 0.0)
    # Add other (scaled by alpha)
    if tl.constexpr(isinstance(other_ptr, tl.tensor)):
        other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        result = selected_x + alpha * other
    else:
        result = selected_x + alpha * other_ptr
    # Apply GELU
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    result_cubed = result * result * result
    tanh_arg = sqrt_2_over_pi * (result + 0.044715 * result_cubed)
    gelu = result * 0.5 * (1.0 + tl.tanh(tanh_arg))
    tl.store(out_ptr + offsets, gelu, mask=mask)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other_scalar = other
        other = torch.tensor(other_scalar, dtype=input.dtype, device=input.device)
    
    # Ensure mask is boolean
    mask = mask.to(torch.bool)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Broadcast other to match input shape
        other = other.expand_as(input)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input tensor")
        if out.device != input.device:
            raise ValueError("Output tensor must be on the same device as input tensor")
    
    # Get total number of elements
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For scalar other, we need to pass it differently
    if not torch.is_tensor(other):
        # Use a different kernel that handles scalar other
        _masked_select_add_gelu_kernel[grid](input, mask, other, out, n, alpha, BLOCK=block)
    else:
        # Use the kernel that handles tensor other
        _masked_select_add_gelu_kernel[grid](input, mask, other, out, n, alpha, BLOCK=block)
    
    return out
import torch
import triton
import triton.language as tl

@triton.jit
def _add_gelu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Add input and scaled other
    z = x + alpha * y
    # Apply GELU approximation
    # For 'none' approximation, use the standard GELU formula
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (z + 0.044715 * x_cubed)
    gelu_result = 0.5 * z * (1.0 + tl.tanh(tanh_arg))
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same shape as input for broadcasting
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    _add_gelu_kernel[grid](input, other, out, n, alpha, BLOCK=block)
    return out

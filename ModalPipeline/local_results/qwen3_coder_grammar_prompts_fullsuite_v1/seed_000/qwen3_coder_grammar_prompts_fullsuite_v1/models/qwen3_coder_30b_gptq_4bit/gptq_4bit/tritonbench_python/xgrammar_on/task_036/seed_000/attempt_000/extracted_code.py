import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh
    y = x * 0.5 * (1.0 + tl.tanh(x * 0.7978845608))  # 0.7978845608 is sqrt(2/pi)
    tl.store(out_ptr + offsets, y, mask=mask)


def add_gelu(input, other, alpha=1, approximate='none', out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast other to match input shape
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    # Add input and scaled other
    temp = input + alpha * other
    
    # Apply GELU
    if out is None:
        out = torch.empty_like(temp)
    else:
        assert out.shape == temp.shape, "Output tensor shape must match input shape"
    
    n = temp.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _gelu_kernel[grid](temp, out, n, BLOCK=block)
    return out
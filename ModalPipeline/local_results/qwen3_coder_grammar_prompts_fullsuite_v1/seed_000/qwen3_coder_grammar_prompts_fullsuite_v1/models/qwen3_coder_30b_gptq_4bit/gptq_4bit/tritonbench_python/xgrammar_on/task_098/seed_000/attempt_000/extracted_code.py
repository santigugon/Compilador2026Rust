import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply GELU
    if approximate == 'none':
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        x_over_sqrt2 = x / tl.sqrt(2.0)
        erf_x_over_sqrt2 = tl.erf(x_over_sqrt2)
        y = x * 0.5 * (1.0 + erf_x_over_sqrt2)
    else:
        # Approximate GELU using tanh
        # GELU approx = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = tl.sqrt(2.0 / tl.pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_x = tl.tanh(tanh_arg)
        y = 0.5 * x * (1.0 + tanh_x)
    
    tl.store(out_ptr + offsets, y, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same device and dtype as input
    other = other.to(input.dtype).to(input.device)
    
    # Broadcast other to match input shape
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    # Perform input - alpha * other
    temp = input - alpha * other
    
    # Output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
        assert out.device == input.device, "Output tensor must be on the same device as input"
    
    # Launch GELU kernel
    n = temp.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    approximate_enum = 'none' if approximate == 'none' else 'tanh'
    _gelu_kernel[grid](temp, out, n, approximate_enum, BLOCK=block)
    
    return out
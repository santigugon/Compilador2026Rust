import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x - alpha * other
    # For simplicity, we assume other is a scalar in this kernel
    # In practice, you'd need to handle broadcasting properly
    
    # GELU computation
    if approximate == 'none':
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        cdf = 0.5 * (1.0 + tl.erf(x / sqrt_2))
        y = x * cdf
    else:
        # Approximate GELU using tanh
        # GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
        y = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, y, mask=mask)


def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    # Handle scalar other
    if not torch.is_tensor(other):
        # For scalar other, we can compute the result directly
        if approximate == 'none':
            # Use PyTorch's built-in GELU for exact computation
            result = torch.gelu(input - alpha * other)
        else:
            # Use PyTorch's approximate GELU
            result = torch.gelu(input - alpha * other, approximate='tanh')
        
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # For tensor other, we need to use Triton
    # First compute input - alpha * other
    if out is not None:
        # If out is provided, we'll compute in-place
        temp = out
        temp.copy_(input - alpha * other)
    else:
        temp = input - alpha * other
    
    # Apply GELU using Triton
    out_tensor = torch.empty_like(temp)
    n = temp.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine approximate mode
    approx_mode = 'none' if approximate == 'none' else 'tanh'
    _gelu_kernel[grid](temp, out_tensor, n, approx_mode, BLOCK=block)
    
    if out is not None:
        out.copy_(out_tensor)
        return out
    return out_tensor
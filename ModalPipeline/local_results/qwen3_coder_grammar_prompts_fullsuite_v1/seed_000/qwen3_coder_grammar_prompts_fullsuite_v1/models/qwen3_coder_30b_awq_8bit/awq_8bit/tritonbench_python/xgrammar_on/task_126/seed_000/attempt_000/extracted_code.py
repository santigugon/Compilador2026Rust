import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Approximate GELU using tanh-based formula
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, y, mask=mask)


def gelu(input, approximate='none'):
    if approximate != 'tanh':
        # For exact GELU, fall back to PyTorch implementation
        return torch.nn.functional.gelu(input, approximate=approximate)
    
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](input, out, n, BLOCK=block)
    return out
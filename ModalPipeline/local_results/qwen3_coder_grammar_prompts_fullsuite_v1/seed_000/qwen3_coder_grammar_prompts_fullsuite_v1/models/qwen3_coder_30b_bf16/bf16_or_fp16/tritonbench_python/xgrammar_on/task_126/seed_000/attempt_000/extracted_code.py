import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == "none":
        # Exact GELU: x * Phi(x) where Phi is the standard normal CDF
        # Using the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = tl.sqrt(2.0 / pi)
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
        y = 0.5 * x * (1.0 + tanh_val)
    else:
        # Approximate GELU using tanh
        pi = 3.141592653589793
        sqrt_2_over_pi = tl.sqrt(2.0 / pi)
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
        y = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, y, mask=mask)

def gelu(input, approximate='none'):
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](input, out, n, approximate, BLOCK=block)
    return out
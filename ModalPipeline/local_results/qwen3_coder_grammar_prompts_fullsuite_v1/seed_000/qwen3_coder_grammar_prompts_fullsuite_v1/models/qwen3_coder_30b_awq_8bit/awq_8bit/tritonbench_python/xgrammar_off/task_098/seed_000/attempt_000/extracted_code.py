import torch
import triton
import triton.language as tl

@triton.jit
def _sub_gelu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Subtract other scaled by alpha from input
    z = x - alpha * y
    # Apply GELU
    if approximate == 'none':
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = z / sqrt_2
        # Approximate erf using the polynomial approximation
        # erf(x) ≈ sign(x) * (1 - exp(-x^2 * (4/pi + a*x^2)/(1 + a*x^2)))
        a = 0.147
        erf_val = tl.where(erf_arg >= 0, 
                          1.0 - tl.exp(-erf_arg * erf_arg * (4.0 / 3.141592653589793 + a * erf_arg * erf_arg) / (1.0 + a * erf_arg * erf_arg)),
                          -1.0 + tl.exp(-erf_arg * erf_arg * (4.0 / 3.141592653589793 + a * erf_arg * erf_arg) / (1.0 + a * erf_arg * erf_arg)))
        gelu_val = 0.5 * z * (1.0 + erf_val)
    else:  # approximate == 'tanh'
        # Approximate GELU using tanh: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        x_cubed = z * z * z
        tanh_arg = sqrt_2_over_pi * (z + 0.044715 * x_cubed)
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
        gelu_val = z * 0.5 * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Expand other to match input shape if needed
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    # Convert approximate to a constexpr value
    approx = approximate
    
    _sub_gelu_kernel[grid](input, other, out, n, alpha, approx, BLOCK=block)
    return out

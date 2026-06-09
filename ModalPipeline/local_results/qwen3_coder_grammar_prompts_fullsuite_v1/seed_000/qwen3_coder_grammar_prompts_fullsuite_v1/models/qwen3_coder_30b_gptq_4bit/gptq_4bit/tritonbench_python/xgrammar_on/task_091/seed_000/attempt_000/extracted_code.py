import torch
import triton
import triton.language as tl

t@triton.jit
def _erfc_sqrt_kernel(x_ptr, out1_ptr, out2_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute erfc
    # Approximation using error function approximation
    # erfc(x) = 1 - erf(x)
    # Using Abramowitz and Stegun approximation for erf
    # For simplicity, using torch's erf and computing erfc as 1 - erf
    # But since we're in Triton, we'll compute both directly
    # Using the relationship: erfc(x) = 1 - erf(x)
    # We'll compute erf using a simple approximation
    # erf(x) ≈ 1 - (1 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5) * exp(-x^2)
    # Constants for approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413788
    a4 = -1.453159026
    a5 = 1.061405429
    
    # Compute erf approximation
    sign = tl.where(x > 0, 1.0, -1.0)
    x_abs = tl.abs(x)
    
    # Calculate t = 1 / (1 + 0.5 * |x|)
    t = 1.0 / (1.0 + 0.5 * x_abs)
    
    # Calculate polynomial
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    
    # erf approximation
    erf_approx = 1.0 - sign * poly * tl.exp(-x_abs * x_abs)
    
    # erfc = 1 - erf
    erfc = 1.0 - erf_approx
    
    # Compute sqrt
    sqrt_x = tl.sqrt(x)
    
    # Store results
    tl.store(out1_ptr + offsets, erfc, mask=mask)
    tl.store(out2_ptr + offsets, sqrt_x, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out1 = torch.empty_like(input)
    out2 = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _erfc_sqrt_kernel[grid](input, out1, out2, n, BLOCK=block)
    return (out1, out2)
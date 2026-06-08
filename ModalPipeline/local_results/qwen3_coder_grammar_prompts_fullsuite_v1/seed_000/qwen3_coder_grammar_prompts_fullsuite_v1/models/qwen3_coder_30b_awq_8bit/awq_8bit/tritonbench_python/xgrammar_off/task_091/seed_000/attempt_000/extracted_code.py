import torch
import triton
import triton.language as tl

@triton.jit
def _erfc_sqrt_kernel(x_ptr, erfc_ptr, sqrt_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute erfc(x) using the approximation: erfc(x) = 1 - erf(x)
    # For numerical stability, we use the relationship with the error function
    # erf(x) can be approximated using a series expansion or lookup tables
    # Here we use a simple approximation for demonstration
    # A more accurate implementation would use a proper erfc approximation
    
    # Approximation for erf(x) using rational function (simplified)
    # This is a basic approximation; a more accurate version would use
    # the full error function implementation
    t = 1.0 / (1.0 + 0.5 * tl.abs(x))
    erf_approx = 1.0 - t * tl.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    
    # erfc(x) = 1 - erf(x)
    erfc = 1.0 - erf_approx
    
    # Compute sqrt(x)
    sqrt_val = tl.sqrt(x)
    
    tl.store(erfc_ptr + offsets, erfc, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_erfc = torch.empty_like(input)
    out_sqrt = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _erfc_sqrt_kernel[grid](input, out_erfc, out_sqrt, n, BLOCK=block)
    return (out_erfc, out_sqrt)

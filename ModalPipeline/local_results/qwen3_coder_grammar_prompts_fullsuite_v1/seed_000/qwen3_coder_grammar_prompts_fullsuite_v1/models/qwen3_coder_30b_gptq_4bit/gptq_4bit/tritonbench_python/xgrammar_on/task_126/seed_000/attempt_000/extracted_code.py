import torch
import triton
import triton.language as tl

def gelu(input, approximate='none'):
    if approximate == 'none':
        # Exact GELU computation
        out = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        @triton.jit
        def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            # GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
            sqrt_2 = 1.4142135623730951
            erf_arg = x / sqrt_2
            # Approximate erf using Taylor series or lookup
            # Using a common approximation for erf
            y = 0.5 * x * (1.0 + tl.erf(erf_arg))
            tl.store(out_ptr + offsets, y, mask=mask)
        
        _gelu_kernel[grid](input, out, n, BLOCK=block)
        return out
    else:
        # Approximate GELU using tanh
        out = torch.empty_like(input)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        @triton.jit
        def _gelu_tanh_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            # Approximate GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            pi = 3.141592653589793
            sqrt_2_over_pi = 1.4142135623730951 / pi
            x_cubed = x * x * x
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            y = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
            tl.store(out_ptr + offsets, y, mask=mask)
        
        _gelu_tanh_kernel[grid](input, out, n, BLOCK=block)
        return out
import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 'none':
        # Standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = x / sqrt_2
        # Approximate erf using polynomial approximation
        # erf(x) ≈ sign(x) * (1 - exp(-x^2 * (4/pi + ax^2) / (1 + ax^2)))
        a = 0.147
        erf_val = tl.where(erf_arg >= 0, 
                          1.0 - tl.exp(-erf_arg * erf_arg * (4.0 / 3.14159 + a * erf_arg * erf_arg) / (1.0 + a * erf_arg * erf_arg)),
                          -1.0 + tl.exp(-erf_arg * erf_arg * (4.0 / 3.14159 + a * erf_arg * erf_arg) / (1.0 + a * erf_arg * erf_arg)))
        y = x * 0.5 * (1.0 + erf_val)
    else:
        # Approximate GELU using tanh
        # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        y = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _std_kernel(x_ptr, out_ptr, mean_ptr, n: tl.constexpr, reduced_n: tl.constexpr, BLOCK: tl.constexpr, keepdim: tl.constexpr, correction: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load mean for this element
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference
    diff = x - mean
    squared_diff = diff * diff
    
    # Store squared differences for reduction
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Apply GELU activation
    input_flat = input.flatten()
    n = input_flat.numel()
    
    # Allocate output for GELU
    gelu_out = torch.empty_like(input_flat)
    
    # Launch GELU kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine approximate parameter
    approx = 'none' if approximate == 'none' else 'tanh'
    
    _gelu_kernel[grid](input_flat, gelu_out, n, BLOCK=block, approximate=approx)
    
    # Reshape to original shape
    gelu_out = gelu_out.view(input.shape)
    
    # Compute standard deviation
    if dim is None:
        # Compute over all dimensions
        if correction == 0:
            std = torch.std(gelu_out, unbiased=False)
        else:
            std = torch.std(gelu_out, unbiased=True)
        if out is not None:
            out.copy_(std)
            return out
        return std
    else:
        # Compute over specified dimensions
        if correction == 0:
            std = torch.std(gelu_out, dim=dim, keepdim=keepdim, unbiased=False)
        else:
            std = torch.std(gelu_out, dim=dim, keepdim=keepdim, unbiased=True)
        if out is not None:
            out.copy_(std)
            return out
        return std

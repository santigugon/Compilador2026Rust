import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 'none':
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = x / sqrt_2
        # Approximate erf using polynomial
        # erf(x) ≈ sign(x) * (1 - exp(-x^2 * (4/π + a*x^2) / (1 + a*x^2)))
        a = 0.147
        exp_arg = -erf_arg * erf_arg * (4.0 / 3.141592653589793 + a * erf_arg * erf_arg)
        erf_val = erf_arg * (1.0 - tl.exp(exp_arg))
        y = x * 0.5 * (1.0 + erf_val)
    else:
        # Approximate GELU: x * sigmoid(1.702 * x)
        y = x * (1.0 / (1.0 + tl.exp(-1.702 * x)))
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _sum_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use atomic add for reduction
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))

@triton.jit
def _mean_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def _var_kernel(x_ptr, mean_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    diff = x - mean
    squared_diff = diff * diff
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

@triton.jit
def _std_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def _reduce_mean_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def _reduce_var_kernel(x_ptr, mean_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    diff = x - mean
    squared_diff = diff * diff
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

@triton.jit
def _reduce_std_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        
    # Apply GELU activation
    gelu_input = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    approx = 'none' if approximate == 'none' else 'tanh'
    _gelu_kernel[grid](input, gelu_input, n, approx, BLOCK=block)
    
    # Compute standard deviation
    if dim is None:
        # Compute over all dimensions
        if correction == 0:
            std = torch.std(gelu_input, unbiased=False)
        else:
            std = torch.std(gelu_input, unbiased=True)
        
        if out is not None:
            out.copy_(std)
            return out
        return std
    else:
        # Compute over specified dimensions
        if correction == 0:
            std = torch.std(gelu_input, dim=dim, keepdim=keepdim, unbiased=False)
        else:
            std = torch.std(gelu_input, dim=dim, keepdim=keepdim, unbiased=True)
        
        if out is not None:
            out.copy_(std)
            return out
        return std
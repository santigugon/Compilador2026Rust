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
        x_over_sqrt2 = x * 0.70710678118654757
        erf_x = tl.erf(x_over_sqrt2)
        y = x * 0.5 * (1.0 + erf_x)
    else:
        # Approximate GELU: x * sigmoid(1.702 * x)
        y = x * (1.0 / (1.0 + tl.exp(-1.702 * x)))
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _std_kernel(x_ptr, mean_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + (offsets // dim_size) % dim_size, mask=mask, other=0.0)
    diff = x - mean
    squared_diff = diff * diff
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Reduce along the last dimension
    reduced_offsets = offsets // dim_size
    tl.atomic_add(out_ptr + reduced_offsets, x, mask=reduced_offsets < (n // dim_size))

@triton.jit
def _mean_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Reduce along the last dimension
    reduced_offsets = offsets // dim_size
    tl.atomic_add(out_ptr + reduced_offsets, x, mask=reduced_offsets < (n // dim_size))
    
    # Normalize by dim_size
    if pid == 0:
        for i in range(n // dim_size):
            tl.store(out_ptr + i, tl.load(out_ptr + i) / dim_size)

@triton.jit
def _std_reduce_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Reduce along the last dimension
    reduced_offsets = offsets // dim_size
    tl.atomic_add(out_ptr + reduced_offsets, x, mask=reduced_offsets < (n // dim_size))

@triton.jit
def _std_final_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Normalize by dim_size
    if pid == 0:
        for i in range(n // dim_size):
            tl.store(out_ptr + i, tl.sqrt(tl.load(out_ptr + i) / dim_size))

@triton.jit
def _std_final_kernel2(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Normalize by (dim_size - correction)
    if pid == 0:
        for i in range(n // dim_size):
            tl.store(out_ptr + i, tl.sqrt(tl.load(out_ptr + i) / (dim_size - 1)))

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    # Apply GELU activation
    gelu_input = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    approximate_val = 'none' if approximate == 'none' else 'tanh'
    _gelu_kernel[grid](input, gelu_input, n, approximate_val, BLOCK=block)
    
    # Compute standard deviation
    if dim is None:
        # Compute over all dimensions
        if correction == 0:
            std = torch.std(gelu_input, unbiased=False)
        else:
            std = torch.std(gelu_input, unbiased=True)
        return std
    else:
        # Compute over specified dimensions
        if not isinstance(dim, tuple):
            dim = (dim,)
        
        # Handle negative dimensions
        dim = tuple(d if d >= 0 else input.dim() + d for d in dim)
        
        # Create output shape
        output_shape = list(input.shape)
        for d in sorted(dim, reverse=True):
            output_shape.pop(d)
        
        if keepdim:
            for d in dim:
                output_shape.insert(d, 1)
        
        # Compute standard deviation
        if correction == 0:
            std = torch.std(gelu_input, dim=dim, keepdim=keepdim, unbiased=False)
        else:
            std = torch.std(gelu_input, dim=dim, keepdim=keepdim, unbiased=True)
        
        return std
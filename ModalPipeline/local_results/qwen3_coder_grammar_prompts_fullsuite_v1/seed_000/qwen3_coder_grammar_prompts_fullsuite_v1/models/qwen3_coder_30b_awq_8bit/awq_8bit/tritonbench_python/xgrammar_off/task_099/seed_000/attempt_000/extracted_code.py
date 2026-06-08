import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 'tanh':
        # GELU approximation using tanh
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    else:
        # Exact GELU using error function
        y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _std_kernel(x_ptr, out_ptr, mean_ptr, n: tl.constexpr, reduced_n: tl.constexpr, BLOCK: tl.constexpr, keepdim: tl.constexpr, correction: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared differences
    diff = x - mean
    squared_diff = diff * diff
    
    # For standard deviation computation, we need to reduce over the specified dimensions
    # This is a simplified version - in practice, this would need more complex reduction logic
    # For now, we'll compute the mean of squared differences and take sqrt
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Handle approximate parameter
    approx = 'none' if approximate == 'none' else 'tanh'
    
    # Apply GELU activation
    input_flat = input.flatten()
    n = input_flat.numel()
    
    # Create output tensor for GELU
    gelu_out = torch.empty_like(input_flat)
    
    # Launch GELU kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](input_flat, gelu_out, n, BLOCK=block, approximate=approx)
    
    # Reshape to original shape
    gelu_out = gelu_out.view(input.shape)
    
    # Compute standard deviation
    if dim is None:
        # Compute over all dimensions
        mean_val = gelu_out.mean()
        squared_diff = (gelu_out - mean_val) ** 2
        var = squared_diff.sum() / (gelu_out.numel() - correction)
        std = torch.sqrt(var)
    else:
        # Compute over specified dimensions
        mean_val = gelu_out.mean(dim=dim, keepdim=True)
        squared_diff = (gelu_out - mean_val) ** 2
        if keepdim:
            var = squared_diff.sum(dim=dim, keepdim=True) / (gelu_out.numel() // mean_val.numel() - correction)
        else:
            var = squared_diff.sum(dim=dim) / (gelu_out.numel() // mean_val.numel() - correction)
        std = torch.sqrt(var)
    
    if out is not None:
        out.copy_(std)
        return out
    
    return std

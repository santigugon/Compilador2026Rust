import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 0:  # exact GELU
        # GELU = x * Φ(x) where Φ is the CDF of the standard normal distribution
        # Using the approximation: GELU = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    else:  # tanh approximation
        # GELU ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(out_ptr + offsets, gelu, mask=mask)

@triton.jit
def _min_kernel(x_ptr, out_ptr, indices_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr, keepdim: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For simplicity, we'll compute the minimum over all elements
    # In a full implementation, this would be more complex for multi-dim reduction
    min_val = tl.min(x)
    min_idx = tl.argmin(x)
    
    tl.store(out_ptr + pid, min_val, mask=pid < dim_size)
    if indices_ptr is not None:
        tl.store(indices_ptr + pid, min_idx, mask=pid < dim_size)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Convert approximate to integer for kernel
    approximate_int = 0 if approximate == 'none' else 1
    
    # Apply GELU
    gelu_input = input.clone()
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create output tensor for GELU
    gelu_out = torch.empty_like(input)
    
    # Apply GELU kernel
    _gelu_kernel[grid](input, gelu_out, n, BLOCK=block, approximate=approximate_int)
    
    # If dim is None, compute minimum over all elements
    if dim is None:
        # Flatten the tensor to compute global minimum
        flat_gelu = gelu_out.flatten()
        min_val = flat_gelu.min()
        min_idx = flat_gelu.argmin()
        
        # Return the minimum value and index
        if out is not None:
            out[0] = min_val
            out[1] = min_idx
            return out
        
        return (min_val, min_idx)
    else:
        # For multi-dimensional case, we need to handle reduction properly
        # This is a simplified version - in practice, you'd want to implement
        # proper reduction kernels for each dimension
        if keepdim:
            # Keep dimensions
            result = gelu_out.min(dim=dim, keepdim=keepdim)
        else:
            # Don't keep dimensions
            result = gelu_out.min(dim=dim, keepdim=keepdim)
        
        if out is not None:
            out[0] = result[0]
            out[1] = result[1]
            return out
        
        return result

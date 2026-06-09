import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU implementation
    if approximate == 'tanh':
        # GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        tanh_arg = 0.7978845608 * (x + 0.044715 * x_cubed)  # sqrt(2/pi) ≈ 0.7978845608
        gelu_x = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    else:
        # Standard GELU: x * Φ(x) where Φ is the CDF of the standard normal distribution
        # Using approximation: x * 0.5 * (1 + tanh(sqrt(π/2) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        tanh_arg = 1.702103772 * (x + 0.044715 * x_cubed)  # sqrt(π/2) ≈ 1.702103772
        gelu_x = x * 0.5 * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(out_ptr + offsets, gelu_x, mask=mask)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Apply GELU activation
    input_flat = input.flatten()
    n = input_flat.numel()
    
    # Allocate output tensor for GELU
    gelu_out = torch.empty_like(input_flat)
    
    # Launch GELU kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](input_flat, gelu_out, n, approximate == 'tanh', BLOCK=block)
    
    # Reshape to original shape
    gelu_out = gelu_out.view(input.shape)
    
    # Compute standard deviation
    if dim is None:
        # Compute over all dimensions
        std_tensor = torch.std(gelu_out, correction=correction)
    else:
        # Compute over specified dimensions
        std_tensor = torch.std(gelu_out, dim=dim, keepdim=keepdim, correction=correction)
    
    # Return result
    if out is not None:
        out.copy_(std_tensor)
        return out
    else:
        return std_tensor

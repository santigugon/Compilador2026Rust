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
    if approximate == 0:  # 'none' mode
        # GELU = x * Phi(x) where Phi is the standard normal CDF
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        term1 = 0.5 * z
        term2 = 1.0 + tl.tanh(sqrt_2_over_pi * (z + 0.044715 * z * z * z))
        result = term1 * term2
    else:  # 'tanh' mode
        # GELU approximation using tanh
        # GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        term1 = 0.5 * z
        term2 = 1.0 + tl.tanh(sqrt_2_over_pi * (z + 0.044715 * z * z * z))
        result = term1 * term2
    tl.store(out_ptr + offsets, result, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same device and dtype as input
    other = other.to(input.device, input.dtype)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
    
    # Get total number of elements
    n = input.numel()
    
    # Set up kernel launch parameters
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine approximate mode
    approx_mode = 0 if approximate == 'none' else 1
    
    # Launch kernel
    _sub_gelu_kernel[grid](input, other, out, n, alpha, approx_mode, BLOCK=block)
    
    return out

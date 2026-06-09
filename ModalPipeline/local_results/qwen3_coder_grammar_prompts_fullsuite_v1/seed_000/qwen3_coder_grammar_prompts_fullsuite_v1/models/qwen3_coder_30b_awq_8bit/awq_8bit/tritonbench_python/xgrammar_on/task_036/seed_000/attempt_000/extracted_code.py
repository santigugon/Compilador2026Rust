import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation
    if approximate == 'tanh':
        # GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    else:
        # Standard GELU: x * Phi(x) where Phi is the standard normal CDF
        # Using approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(out_ptr + offsets, gelu, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast tensors
    input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Compute the sum
    sum_tensor = input + alpha * other
    
    # Apply GELU
    n = sum_tensor.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _gelu_kernel[grid](sum_tensor, out, n, BLOCK=block, approximate=approximate)
    
    return out
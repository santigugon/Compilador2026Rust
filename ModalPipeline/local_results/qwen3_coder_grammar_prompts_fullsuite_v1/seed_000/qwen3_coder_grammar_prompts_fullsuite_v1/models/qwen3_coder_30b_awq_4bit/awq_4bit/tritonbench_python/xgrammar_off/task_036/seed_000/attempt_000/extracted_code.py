import torch
import triton
import triton.language as tl

@triton.jit
def _add_gelu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Compute x + alpha * y
    sum_val = x + alpha * y
    # Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using approximation: 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    tanh_arg = 0.7978845608 * (sum_val + 0.044715 * x_cubed)
    tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
    gelu_val = 0.5 * sum_val * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if not torch.is_tensor(other):
        # Handle scalar case
        other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
        other_ptr = other_tensor
    else:
        other_ptr = other
    
    _add_gelu_kernel[grid](input, other_ptr, out, n, alpha, BLOCK=block)
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    approximate,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # GELU approximation
    if approximate == "tanh":
        # GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = input
        x_cubed = x * x * x
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = tl.tanh(tanh_arg)
        output = 0.5 * x * (1.0 + tanh_val)
    else:
        # Standard GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
        x = input
        sqrt_2 = 1.4142135623730951
        erf_arg = x / sqrt_2
        erf_val = tl.erf(erf_arg)
        output = 0.5 * x * (1.0 + erf_val)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Handle scalar addition
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast other to match input shape
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    # Add tensors
    input_plus_other = input + alpha * other
    
    # Apply GELU
    n_elements = input_plus_other.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    gelu_kernel[grid](
        input_plus_other,
        out,
        n_elements,
        approximate,
        BLOCK_SIZE
    )
    
    return out

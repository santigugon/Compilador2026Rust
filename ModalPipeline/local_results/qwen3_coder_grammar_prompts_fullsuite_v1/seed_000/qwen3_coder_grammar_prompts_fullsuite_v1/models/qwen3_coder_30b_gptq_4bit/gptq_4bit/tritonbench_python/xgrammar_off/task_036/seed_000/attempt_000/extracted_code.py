import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    approximate,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    scaled_other = other * alpha
    combined = input + scaled_other
    
    # GELU computation
    if approximate == 'none':
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = combined / sqrt_2
        erf_val = tl.math.erf(erf_arg)
        gelu_result = 0.5 * combined * (1.0 + erf_val)
    else:
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi_inv = 0.3183098861837907
        sqrt_2_pi = 1.4142135623730951 * 0.3183098861837907
        x_cubed = combined * combined * combined
        tanh_arg = sqrt_2_pi * (combined + 0.044715 * x_cubed)
        tanh_val = tl.math.tanh(tanh_arg)
        gelu_result = 0.5 * combined * (1.0 + tanh_val)
    
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and other have compatible shapes for element-wise addition
    if isinstance(other, (int, float)):
        other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other_tensor = other
    
    # Expand other_tensor to match input shape if needed
    if other_tensor.shape != input.shape:
        other_tensor = other_tensor.expand_as(input)
    
    # Ensure all tensors are on the same device
    other_tensor = other_tensor.to(input.device)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    gelu_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other_tensor.data_ptr(),
        output_ptr=out.data_ptr(),
        alpha=alpha,
        approximate=approximate,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

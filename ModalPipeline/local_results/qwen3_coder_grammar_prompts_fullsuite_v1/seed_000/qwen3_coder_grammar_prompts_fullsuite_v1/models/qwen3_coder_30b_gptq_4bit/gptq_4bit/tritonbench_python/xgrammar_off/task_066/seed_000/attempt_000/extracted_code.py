import torch
import triton
import triton.language as tl

@triton.jit
def _fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, out_ptr,
    n: tl.constexpr, alpha: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0)
    
    # Select elements based on mask
    selected = tl.where(mask_val, input, 0.0)
    
    # Add other (scaled by alpha)
    if other_ptr is not None:
        other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        result = selected + alpha * other
    else:
        result = selected + alpha
    
    # Apply GELU activation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x^2)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x = result
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * x * (1 + 0.044715 * x_cubed)
    gelu_result = 0.5 * x * (1 + tl.tanh(tanh_arg))
    
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same shape as input for broadcasting
    if other.shape != input.shape:
        other = other.expand_as(input)
    
    # Ensure mask has the same shape as input
    if mask.shape != input.shape:
        mask = mask.expand_as(input)
    
    # Create output tensor
    if out is not None:
        out = torch.empty_like(out)
    else:
        out = torch.empty_like(input)
    
    # Flatten tensors for processing
    input_flat = input.flatten()
    mask_flat = mask.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    n = input_flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle case where other is a scalar
    if not torch.is_tensor(other):
        other_ptr = None
    else:
        other_ptr = other_flat.data_ptr()
    
    _fused_masked_select_add_gelu_kernel[grid](
        input_flat.data_ptr(),
        mask_flat.data_ptr(),
        other_ptr,
        out_flat.data_ptr(),
        n,
        alpha,
        BLOCK=block
    )
    
    return out

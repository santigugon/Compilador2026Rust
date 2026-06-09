import torch
import triton
import triton.language as tl

def gelu(x):
    # Using the standard GELU approximation
    return 0.5 * x * (1 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

@triton.jit
def _fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, out_ptr,
    n: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Select elements based on mask
    selected = tl.where(mask_val, input, 0.0)
    
    # Add other (scaled by alpha)
    if tl.is_scalar(other_ptr):
        other = other_ptr
    else:
        other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    result = selected + alpha * other
    
    # Apply GELU
    gelu_result = gelu(result)
    
    tl.store(out_ptr + offsets, gelu_result, mask=mask)


def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    # Validate inputs
    if input.shape != mask.shape:
        raise ValueError("input and mask must have the same shape")
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same shape as input
    if other.shape != input.shape:
        # Broadcast other to match input shape
        other = other.expand_as(input)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
    
    # Flatten tensors for kernel processing
    n = input.numel()
    
    # Determine block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel
    _fused_masked_select_add_gelu_kernel[grid](
        input, mask, other, out,
        n, alpha, BLOCK=block
    )
    
    return out
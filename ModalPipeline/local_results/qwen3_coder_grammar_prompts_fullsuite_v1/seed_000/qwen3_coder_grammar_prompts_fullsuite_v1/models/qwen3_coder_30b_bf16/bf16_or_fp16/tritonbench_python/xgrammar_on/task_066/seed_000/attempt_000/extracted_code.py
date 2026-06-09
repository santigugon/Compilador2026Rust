import torch
import triton
import triton.language as tl

def gelu(x):
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using a more numerically stable implementation
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))

@triton.jit
def _fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, out_ptr,
    n: tl.constexpr,
    alpha: tl.constexpr,
    input_stride: tl.constexpr,
    mask_stride: tl.constexpr,
    other_stride: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    
    # Load mask
    mask_vals = tl.load(mask_ptr + offsets * mask_stride, mask=mask, other=False)
    
    # Load other values (could be scalar or tensor)
    other_vals = tl.load(other_ptr + offsets * other_stride, mask=mask, other=0.0)
    
    # Apply alpha scaling
    other_vals = alpha * other_vals
    
    # Select based on mask
    selected_vals = tl.where(mask_vals, input_vals + other_vals, 0.0)
    
    # Apply GELU
    gelu_vals = gelu(selected_vals)
    
    # Store result
    tl.store(out_ptr + offsets * out_stride, gelu_vals, mask=mask)

@triton.jit
def _fused_masked_select_add_gelu_kernel_scalar(
    input_ptr, mask_ptr, other_scalar, out_ptr,
    n: tl.constexpr,
    alpha: tl.constexpr,
    input_stride: tl.constexpr,
    mask_stride: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    
    # Load mask
    mask_vals = tl.load(mask_ptr + offsets * mask_stride, mask=mask, other=False)
    
    # Apply alpha scaling with scalar
    other_vals = alpha * other_scalar
    
    # Select based on mask
    selected_vals = tl.where(mask_vals, input_vals + other_vals, 0.0)
    
    # Apply GELU
    gelu_vals = gelu(selected_vals)
    
    # Store result
    tl.store(out_ptr + offsets * out_stride, gelu_vals, mask=mask)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        # For scalar other, we can use a simpler kernel
        if out is None:
            out = torch.empty_like(input)
        
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        _fused_masked_select_add_gelu_kernel_scalar[grid](
            input, mask, other, out,
            n, alpha,
            input.stride(0) if input.dim() > 0 else 1,
            mask.stride(0) if mask.dim() > 0 else 1,
            out.stride(0) if out.dim() > 0 else 1,
            BLOCK=block
        )
        return out
    
    # For tensor other case
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure all tensors have the same shape for broadcasting
    input = input.contiguous()
    mask = mask.contiguous()
    other = other.contiguous()
    out = out.contiguous()
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _fused_masked_select_add_gelu_kernel[grid](
        input, mask, other, out,
        n, alpha,
        input.stride(0) if input.dim() > 0 else 1,
        mask.stride(0) if mask.dim() > 0 else 1,
        other.stride(0) if other.dim() > 0 else 1,
        out.stride(0) if out.dim() > 0 else 1,
        BLOCK=block
    )
    return out
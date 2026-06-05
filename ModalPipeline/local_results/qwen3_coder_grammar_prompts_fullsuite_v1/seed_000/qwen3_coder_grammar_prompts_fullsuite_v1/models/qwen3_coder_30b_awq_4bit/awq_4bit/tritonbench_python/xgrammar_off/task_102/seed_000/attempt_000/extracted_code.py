import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_mul_kernel(
    input_ptr, other_ptr, out_ptr,
    input_stride, other_stride, out_stride,
    dim_size: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
    is_other_tensor: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < numel
    
    # Load input
    input_offsets = offsets
    input = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute softmax
    # For numerical stability, subtract the max
    max_val = tl.max(input, axis=0)
    exp_input = tl.exp(input - max_val)
    sum_exp = tl.sum(exp_input, axis=0)
    softmax = exp_input / sum_exp
    
    # Load other (either tensor or scalar)
    if is_other_tensor:
        other_offsets = offsets
        other = tl.load(other_ptr + other_offsets, mask=mask, other=0.0)
    else:
        other = tl.load(other_ptr, mask=mask, other=0.0)
    
    # Compute output
    out = softmax * other
    tl.store(out_ptr + input_offsets, out, mask=mask)

def softmax_mul(input, other, dim, dtype=None, out=None):
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
        if torch.is_tensor(other):
            other = other.to(dtype)
    
    # Handle out parameter
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
    
    # Handle scalar other
    is_other_scalar = not torch.is_tensor(other)
    if is_other_scalar:
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Get the size of the specified dimension
    dim_size = input.size(dim)
    numel = input.numel()
    
    # Create a block size
    BLOCK = 256
    
    # Compute grid size
    grid = (triton.cdiv(numel, BLOCK),)
    
    # Prepare strides
    input_stride = input.stride()
    other_stride = other.stride() if torch.is_tensor(other) else 0
    out_stride = out.stride()
    
    # Launch kernel
    _softmax_mul_kernel[grid](
        input, other, out,
        input_stride[0] if len(input_stride) > 0 else 0,
        other_stride if torch.is_tensor(other) else 0,
        out_stride[0] if len(out_stride) > 0 else 0,
        dim_size,
        numel,
        BLOCK,
        is_other_tensor=torch.is_tensor(other)
    )
    
    return out

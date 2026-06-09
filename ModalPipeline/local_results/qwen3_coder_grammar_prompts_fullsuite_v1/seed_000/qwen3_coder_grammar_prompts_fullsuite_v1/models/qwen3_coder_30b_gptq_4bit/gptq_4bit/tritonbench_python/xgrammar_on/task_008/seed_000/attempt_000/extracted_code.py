import torch
import triton
import triton.language as tl

def _sub_kernel(input_ptr, other_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    result = input - alpha * other
    tl.store(out_ptr + offsets, result, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same dtype as input
    if other.dtype != input.dtype:
        other = other.to(input.dtype)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input_expanded, other_expanded = torch.broadcast_tensors(input, other)
        out = torch.empty_like(input_expanded)
        n = input_expanded.numel()
    else:
        out = torch.empty_like(input)
        n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Convert to contiguous tensors for kernel
    input_contiguous = input.contiguous()
    other_contiguous = other.contiguous()
    out_contiguous = out.contiguous()
    
    _sub_kernel[grid](input_contiguous, other_contiguous, out_contiguous, n, alpha, BLOCK=block)
    
    return out_contiguous
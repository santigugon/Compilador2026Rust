import torch
import triton
import triton.language as tl

@triton.jit
def _add_mean_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty_like(input)
    
    # Compute the elementwise addition
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _add_mean_kernel[grid](input, other, output, n, alpha, BLOCK=block)
    
    # Compute mean along specified dimension
    if dim is None:
        # Compute mean over all elements
        return output.mean(keepdim=keepdim)
    else:
        # Compute mean along specified dimension(s)
        return output.mean(dim=dim, keepdim=keepdim)

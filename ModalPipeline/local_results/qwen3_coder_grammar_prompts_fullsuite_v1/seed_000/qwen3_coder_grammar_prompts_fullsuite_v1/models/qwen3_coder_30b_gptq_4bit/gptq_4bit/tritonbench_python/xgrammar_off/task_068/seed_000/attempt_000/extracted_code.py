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
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
        if torch.is_tensor(other):
            other = other.to(dtype)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast other to match input shape
    other = torch.broadcast_tensors(input, other)[1]
    
    # Perform elementwise addition
    if out is not None:
        output = out
    else:
        output = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _add_mean_kernel[grid](input, other, output, n, alpha, BLOCK=block)
    
    # Compute mean along specified dimension
    if dim is None:
        # Compute mean over all elements
        result = torch.mean(output, keepdim=keepdim)
    else:
        # Compute mean along specified dimension(s)
        result = torch.mean(output, dim=dim, keepdim=keepdim)
    
    # If out parameter is provided, copy result to out tensor
    if out is not None:
        out.copy_(result)
        return out
    
    return result

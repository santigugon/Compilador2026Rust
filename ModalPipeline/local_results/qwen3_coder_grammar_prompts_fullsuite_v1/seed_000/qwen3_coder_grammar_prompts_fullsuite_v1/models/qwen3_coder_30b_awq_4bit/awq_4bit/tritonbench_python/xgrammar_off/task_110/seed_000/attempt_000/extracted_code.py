import torch
import triton
import triton.language as tl

@triton.jit
def _exp_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _sum_kernel(x_ptr, out_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use tl.sum to compute the sum
    sum_val = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, sum_val, mask=pid < 1)

@triton.jit
def _mean_kernel(x_ptr, out_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use tl.sum to compute the sum
    sum_val = tl.sum(x, axis=0)
    mean_val = sum_val / size
    tl.store(out_ptr + pid, mean_val, mask=pid < 1)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        dim = 0 if dim is None else dim
        keepdim = True
    
    # If no dim is specified, compute mean over all elements
    if dim is None:
        # Compute exp
        exp_out = torch.empty_like(input, dtype=torch.float32)
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _exp_kernel[grid](input, exp_out, n, BLOCK=block)
        
        # Compute mean
        sum_out = torch.empty(1, dtype=torch.float32)
        _sum_kernel[1](exp_out, sum_out, n, BLOCK=block)
        mean_val = sum_out[0] / n
        
        # Return result
        result = torch.tensor(mean_val, dtype=torch.float32)
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # If dim is specified, compute mean along that dimension
    # First compute exp
    exp_out = torch.empty_like(input, dtype=torch.float32)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _exp_kernel[grid](input, exp_out, n, BLOCK=block)
    
    # Compute mean along specified dimension
    result = torch.mean(exp_out, dim=dim, keepdim=keepdim)
    
    if out is not None:
        out.copy_(result)
        return out
    return result

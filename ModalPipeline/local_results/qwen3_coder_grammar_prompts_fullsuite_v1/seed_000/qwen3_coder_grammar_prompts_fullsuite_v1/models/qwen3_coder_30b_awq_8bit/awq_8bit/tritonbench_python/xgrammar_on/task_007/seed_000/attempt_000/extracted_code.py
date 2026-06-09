import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _argmax_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # For simplicity, we assume a 1D reduction
    # In practice, this would need more complex indexing for multi-dim
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    # Simple argmax implementation for demonstration
    # This is a simplified version - full implementation would require
    # more complex reduction logic
    max_val = tl.max(x)
    max_idx = tl.argmin(x)
    tl.store(out_ptr + pid, max_idx, mask=pid < 1)


def sigmoid_argmax(input, dim=None, keepdim=False):
    # Apply sigmoid
    sigmoid_input = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sigmoid_kernel[grid](input, sigmoid_input, n, BLOCK=block)
    
    # Apply argmax
    if dim is None:
        # Flatten and find argmax
        flat_input = sigmoid_input.flatten()
        return torch.argmax(flat_input)
    else:
        # Find argmax along specified dimension
        return torch.argmax(sigmoid_input, dim=dim, keepdim=keepdim)

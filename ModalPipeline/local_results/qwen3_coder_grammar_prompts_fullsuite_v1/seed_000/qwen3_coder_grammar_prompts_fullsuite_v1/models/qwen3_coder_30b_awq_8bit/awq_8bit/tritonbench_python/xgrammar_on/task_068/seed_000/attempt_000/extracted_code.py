import torch
import triton
import triton.language as tl

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
        if torch.is_tensor(other):
            other = other.to(dtype)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast tensors
    input, other = torch.broadcast_tensors(input, other)
    
    # Compute the sum of (input + alpha * other)
    if dim is None:
        # Compute mean over all elements
        result = (input + alpha * other).mean()
    else:
        # Compute mean along specified dimension(s)
        result = (input + alpha * other).mean(dim=dim, keepdim=keepdim)
    
    # Handle output tensor
    if out is not None:
        out.copy_(result)
        return out
    
    return result
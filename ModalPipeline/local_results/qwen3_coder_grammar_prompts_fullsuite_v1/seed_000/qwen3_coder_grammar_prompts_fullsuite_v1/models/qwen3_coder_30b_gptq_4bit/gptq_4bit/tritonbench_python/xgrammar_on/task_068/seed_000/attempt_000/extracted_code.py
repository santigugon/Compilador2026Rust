import torch
import triton
import triton.language as tl

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype) if torch.is_tensor(other) else other
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Add the tensors
    if alpha != 1:
        other = other * alpha
    
    # Broadcast and add
    input = input + other
    
    # Compute mean
    if dim is None:
        # Compute mean over all elements
        result = input.mean(keepdim=keepdim)
    else:
        # Compute mean along specified dimension(s)
        result = input.mean(dim=dim, keepdim=keepdim)
    
    # Handle output tensor
    if out is not None:
        out.copy_(result)
        return out
    
    return result
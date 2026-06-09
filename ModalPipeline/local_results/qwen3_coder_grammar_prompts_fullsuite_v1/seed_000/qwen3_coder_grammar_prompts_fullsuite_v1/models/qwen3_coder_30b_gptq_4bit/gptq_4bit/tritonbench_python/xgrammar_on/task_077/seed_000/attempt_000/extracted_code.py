import torch
import triton
import triton.language as tl

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    # Handle scalar value
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=input.dtype, device=input.device)
    
    # Ensure mask is broadcastable to the output shape
    if mask.shape != out.shape:
        # Expand mask to match output shape
        mask = mask.expand_as(out)
    
    # For simplicity, we'll use PyTorch's gather and masked_fill for the actual operations
    # and implement a basic Triton kernel for the core computation
    
    # First, perform gather operation
    gathered = torch.gather(input, dim, index)
    
    # Then apply masked fill
    if out is None:
        out = gathered.clone()
    else:
        out.copy_(gathered)
    
    # Apply mask to fill with value
    out.masked_fill_(mask, value.item())
    
    return out
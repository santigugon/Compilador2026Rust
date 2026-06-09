import torch
import triton
import triton.language as tl

def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Handle scalar inputs
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    # Get input dimensions
    batch_size = x.shape[0]
    channel_size = x.shape[1]
    
    # Flatten spatial dimensions
    spatial_size = x.shape[2] * x.shape[3]
    x_flat = x.view(batch_size, channel_size, spatial_size)
    
    # Output tensor
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
    
    # Handle weight and bias
    if weight is not None:
        weight = weight.view(1, channel_size, 1)
    else:
        weight = torch.ones(1, channel_size, 1, device=x.device, dtype=x.dtype)
    
    if bias is not None:
        bias = bias.view(1, channel_size, 1)
    else:
        bias = torch.zeros(1, channel_size, 1, device=x.device, dtype=x.dtype)
    
    # Update running statistics if training
    if training:
        # Compute batch statistics
        batch_mean = x_flat.mean(dim=(0, 2))
        batch_var = x_flat.var(dim=(0, 2), unbiased=False)
        
        # Update running statistics
        running_mean.data = (1 - momentum) * running_mean + momentum * batch_mean
        running_var.data = (1 - momentum) * running_var + momentum * batch_var
    
    # Get batch statistics
    if training:
        batch_mean = x_flat.mean(dim=(0, 2))
        batch_var = x_flat.var(dim=(0, 2), unbiased=False)
        # Use batch statistics
        mean = batch_mean.view(1, channel_size, 1)
        var = batch_var.view(1, channel_size, 1)
    else:
        # Use running statistics
        mean = running_mean.view(1, channel_size, 1)
        var = running_var.view(1, channel_size, 1)
    
    # Apply batch normalization
    x_normalized = (x_flat - mean) / (torch.sqrt(var + eps))
    
    # Apply affine transformation
    x_normalized = x_normalized * weight + bias
    
    # Apply Hardsigmoid activation
    # Hardsigmoid: min(1, max(0, (x + 3) / 6))
    x_hardsigmoid = torch.clamp((x_normalized + 3.0) / 6.0, 0.0, 1.0)
    
    # Reshape back to original shape
    out = x_hardsigmoid.view(batch_size, channel_size, x.shape[2], x.shape[3])
    
    return out
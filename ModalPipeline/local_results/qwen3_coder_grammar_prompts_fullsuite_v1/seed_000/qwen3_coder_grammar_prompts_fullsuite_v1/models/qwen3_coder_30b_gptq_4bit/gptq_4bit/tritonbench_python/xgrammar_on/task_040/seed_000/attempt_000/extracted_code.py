import torch
import triton
import triton.language as tl

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    N, C = input.shape[0], input.shape[1]
    L = input.shape[2] if len(input.shape) > 2 else 1
    
    # Flatten input for easier processing
    input_flat = input.view(N * C, L)
    
    # Output tensor
    out = torch.empty_like(input)
    out_flat = out.view(N * C, L)
    
    # Handle weight and bias
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Compute batch statistics
    if training:
        # Compute mean and variance for each channel
        mean = input_flat.mean(dim=1)
        var = input_flat.var(dim=1, unbiased=False)
        
        # Update running statistics
        running_mean.copy_(running_mean * (1 - momentum) + mean * momentum)
        running_var.copy_(running_var * (1 - momentum) + var * momentum)
        
        # Normalize
        x_norm = (input_flat - mean.unsqueeze(1)) / (var.unsqueeze(1) + eps).sqrt()
    else:
        # Use running statistics
        x_norm = (input_flat - running_mean.unsqueeze(1)) / (running_var.unsqueeze(1) + eps).sqrt()
    
    # Apply weight and bias if provided
    if weight is not None and bias is not None:
        x_norm = x_norm * weight.unsqueeze(1) + bias.unsqueeze(1)
    elif weight is not None:
        x_norm = x_norm * weight.unsqueeze(1)
    elif bias is not None:
        x_norm = x_norm + bias.unsqueeze(1)
    
    # Apply sigmoid
    out_flat = 1.0 / (1.0 + torch.exp(-x_norm))
    
    # Reshape back to original shape
    return out.view(input.shape)
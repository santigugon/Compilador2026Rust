import torch
import triton
import triton.language as tl

def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Handle scalar inputs
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    oH = (iH + 2 * padding[0] - (dilation[0] * (kH - 1) + 1)) // stride[0] + 1
    oW = (iW + 2 * padding[1] - (dilation[1] * (kW - 1) + 1)) // stride[1] + 1
    
    # Initialize output tensor
    if inplace:
        output = input
    else:
        output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    
    # Perform convolution
    conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Perform batch normalization
    if training:
        # Compute batch statistics
        mean = conv_output.mean(dim=(0, 2, 3), keepdim=True)
        var = conv_output.var(dim=(0, 2, 3), keepdim=True)
        
        # Update running statistics
        if running_mean is not None:
            running_mean = (1 - momentum) * running_mean + momentum * mean
        if running_var is not None:
            running_var = (1 - momentum) * running_var + momentum * var
        
        # Normalize
        normalized = (conv_output - mean) / torch.sqrt(var + eps)
    else:
        # Use running statistics
        if running_mean is not None and running_var is not None:
            normalized = (conv_output - running_mean) / torch.sqrt(running_var + eps)
        else:
            # Fallback to batch statistics if running stats not provided
            mean = conv_output.mean(dim=(0, 2, 3), keepdim=True)
            var = conv_output.var(dim=(0, 2, 3), keepdim=True)
            normalized = (conv_output - mean) / torch.sqrt(var + eps)
    
    # Apply batch normalization parameters
    if bn_weight is not None:
        normalized = normalized * bn_weight.view(1, -1, 1, 1)
    if bn_bias is not None:
        normalized = normalized + bn_bias.view(1, -1, 1, 1)
    
    # Apply ReLU
    if inplace:
        output = torch.nn.functional.relu_(normalized)
    else:
        output = torch.nn.functional.relu(normalized)
    
    return output
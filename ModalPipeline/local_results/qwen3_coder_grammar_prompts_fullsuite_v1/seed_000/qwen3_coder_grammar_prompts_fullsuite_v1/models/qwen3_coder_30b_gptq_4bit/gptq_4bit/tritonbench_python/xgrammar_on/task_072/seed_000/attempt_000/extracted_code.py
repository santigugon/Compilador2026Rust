import torch
import triton
import triton.language as tl

def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Handle scalar stride, padding, and dilation
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
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)

    # Perform convolution
    conv_output = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # Perform batch normalization
    if training:
        # Compute batch statistics
        batch_mean = conv_output.mean(dim=(0, 2, 3))
        batch_var = conv_output.var(dim=(0, 2, 3), unbiased=False)
        
        # Update running statistics
        if running_mean is not None:
            running_mean.copy_(batch_mean * momentum + running_mean * (1 - momentum))
        if running_var is not None:
            running_var.copy_(batch_var * momentum + running_var * (1 - momentum))
        
        # Normalize
        normalized = (conv_output - batch_mean[None, :, None, None]) / (batch_var[None, :, None, None] + eps).sqrt()
    else:
        # Use running statistics
        if running_mean is not None and running_var is not None:
            normalized = (conv_output - running_mean[None, :, None, None]) / (running_var[None, :, None, None] + eps).sqrt()
        else:
            normalized = conv_output

    # Apply batch normalization scaling and shifting
    if bn_weight is not None and bn_bias is not None:
        normalized = normalized * bn_weight[None, :, None, None] + bn_bias[None, :, None, None]

    # Apply ReLU
    if inplace:
        normalized.relu_(inplace=True)
        return normalized
    else:
        return normalized.relu()
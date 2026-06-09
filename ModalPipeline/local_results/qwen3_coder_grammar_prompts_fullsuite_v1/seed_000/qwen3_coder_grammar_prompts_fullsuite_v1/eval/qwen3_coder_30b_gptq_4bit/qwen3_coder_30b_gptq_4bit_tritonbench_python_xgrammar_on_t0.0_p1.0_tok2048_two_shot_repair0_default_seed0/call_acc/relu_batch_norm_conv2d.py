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
##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import nn

import torch
from torch import nn

# Define a simple test function
def test_relu_batch_norm_conv2d():
    # Define input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(4, 3, 32, 32)  # Example: 4 images, 3 channels, 32x32 resolution
    
    # Define convolution weight tensor (out_channels, in_channels/groups, kernel_height, kernel_width)
    weight_tensor = torch.randn(6, 3, 3, 3)  # Example: 6 filters, 3 input channels, 3x3 kernel
    
    # Define optional bias tensor (out_channels)
    bias_tensor = torch.randn(6)  # Example: bias for each of the 6 filters
    
    # Define batch normalization parameters
    running_mean = torch.zeros(6)
    running_var = torch.ones(6)
    bn_weight = torch.ones(6)
    bn_bias = torch.zeros(6)
    
    # Call the relu_batch_norm_conv2d function
    output_tensor = relu_batch_norm_conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        running_mean=running_mean,
        running_var=running_var,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
        inplace=False
    )

    # Print the shape of the output tensor
    print(f"Output tensor shape: {output_tensor.shape}")
    
    # Check if output tensor has the expected shape
    expected_shape = (4, 6, 32, 32)  # 4 images, 6 output channels, 32x32 resolution
    assert output_tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {output_tensor.shape}"

    return output_tensor

# Run the test
output = test_relu_batch_norm_conv2d()
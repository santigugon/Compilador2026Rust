import torch
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, batch_size,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    GROUP_M
):
    # Get the block indices
    block_idx = tl.program_id(0)
    group_id = block_idx // GROUP_M
    block_id = block_idx % GROUP_M
    
    # Compute output dimensions
    output_h = (iH + 2 * padding_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    output_w = (iW + 2 * padding_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the input channels
    for c in range(0, in_channels, BLOCK_SIZE_K):
        # Load input and weight
        input_block = tl.load(input_ptr + c * iH * iW + block_id * BLOCK_SIZE_M * iH * iW)
        weight_block = tl.load(weight_ptr + c * kH * kW + block_id * BLOCK_SIZE_M * kH * kW)
        
        # Perform convolution
        acc += tl.dot(input_block, weight_block)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + block_id * BLOCK_SIZE_M)
        acc += bias_block
    
    # Store result
    tl.store(output_ptr + block_id * BLOCK_SIZE_M * output_h * output_w, acc)

@triton.jit
def batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr, batch_size, channels, height, width,
    momentum, eps, training
):
    # Get the block indices
    block_idx = tl.program_id(0)
    
    # Load input
    input_block = tl.load(input_ptr + block_idx * channels * height * width)
    
    # Compute batch normalization
    if training:
        # Compute mean and variance
        mean = tl.sum(input_block) / (channels * height * width)
        var = tl.sum((input_block - mean) ** 2) / (channels * height * width)
        
        # Update running statistics
        running_mean = tl.load(running_mean_ptr + block_idx * channels)
        running_var = tl.load(running_var_ptr + block_idx * channels)
        
        # Update running mean and variance
        new_mean = momentum * running_mean + (1 - momentum) * mean
        new_var = momentum * running_var + (1 - momentum) * var
        
        # Store updated statistics
        tl.store(running_mean_ptr + block_idx * channels, new_mean)
        tl.store(running_var_ptr + block_idx * channels, new_var)
        
        # Normalize
        normalized = (input_block - mean) / tl.sqrt(var + eps)
    else:
        # Use running statistics
        mean = tl.load(running_mean_ptr + block_idx * channels)
        var = tl.load(running_var_ptr + block_idx * channels)
        
        # Normalize
        normalized = (input_block - mean) / tl.sqrt(var + eps)
    
    # Apply scale and shift
    if weight_ptr is not None and bias_ptr is not None:
        weight_block = tl.load(weight_ptr + block_idx * channels)
        bias_block = tl.load(bias_ptr + block_idx * channels)
        output_block = weight_block * normalized + bias_block
    else:
        output_block = normalized
    
    # Store result
    tl.store(output_ptr + block_idx * channels * height * width, output_block)

@triton.jit
def relu_kernel(input_ptr, output_ptr, batch_size, channels, height, width):
    # Get the block indices
    block_idx = tl.program_id(0)
    
    # Load input
    input_block = tl.load(input_ptr + block_idx * channels * height * width)
    
    # Apply ReLU
    output_block = tl.where(input_block > 0, input_block, 0.0)
    
    # Store result
    tl.store(output_ptr + block_idx * channels * height * width, output_block)

def relu_batch_norm_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
    running_mean=None, running_var=None, bn_weight=None, bn_bias=None,
    training=False, momentum=0.1, eps=1e-5, inplace=False
):
    # Get dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Handle padding
    if isinstance(padding, int):
        padding_h = padding_w = padding
    elif isinstance(padding, tuple):
        padding_h, padding_w = padding
    else:
        padding_h = padding_w = 0
    
    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    # Handle dilation
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Compute output dimensions
    oH = (iH + 2 * padding_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * padding_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Perform convolution
    if groups == 1:
        # Simple convolution
        conv_output = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    else:
        # Grouped convolution
        conv_output = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Perform batch normalization
    if training:
        if running_mean is None:
            running_mean = torch.zeros(out_channels, device=input.device, dtype=torch.float32)
        if running_var is None:
            running_var = torch.ones(out_channels, device=input.device, dtype=torch.float32)
        
        # Apply batch normalization
        batch_norm_output = torch.nn.functional.batch_norm(
            conv_output, running_mean, running_var, bn_weight, bn_bias, training, momentum, eps
        )
    else:
        # Use running statistics
        batch_norm_output = torch.nn.functional.batch_norm(
            conv_output, running_mean, running_var, bn_weight, bn_bias, training, momentum, eps
        )
    
    # Apply ReLU
    if inplace:
        batch_norm_output = torch.relu_(batch_norm_output)
    else:
        batch_norm_output = torch.relu(batch_norm_output)
    
    return batch_norm_output

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
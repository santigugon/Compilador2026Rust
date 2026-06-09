import torch
import triton
import triton.language as tl
import math

@triton.jit
def _leaky_relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width, output_height, output_width,
    weight_height, weight_width, channels, num_filters,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, negative_slope,
    input_batch, BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    filter_idx = tl.program_id(1)
    
    # Calculate which group this filter belongs to
    group_idx = filter_idx // (num_filters // groups)
    
    # Initialize output accumulator
    output_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels and spatial dimensions
    for c in range(channels):
        # Check if this channel belongs to current group
        if c // (channels // groups) != group_idx:
            continue
            
        for kh in range(weight_height):
            for kw in range(weight_width):
                # Calculate input position with dilation and padding
                ih = tl.arange(0, BLOCK_SIZE) // output_width * stride_h - padding_h + kh * dilation_h
                iw = tl.arange(0, BLOCK_SIZE) % output_width * stride_w - padding_w + kw * dilation_w
                
                # Calculate input indices
                input_idx = batch_idx * (input_height * input_width * channels) + \
                           ih * (input_width * channels) + \
                           iw * channels + c
                
                # Calculate weight index
                weight_idx = filter_idx * (weight_height * weight_width * channels) + \
                            kh * (weight_width * channels) + \
                            kw * channels + c
                
                # Load input and weight values
                input_val = tl.load(input_ptr + input_idx, mask=(ih >= 0) & (ih < input_height) & 
                                   (iw >= 0) & (iw < input_width), other=0.0)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Accumulate convolution result
                output_val += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_idx = filter_idx
        bias_val = tl.load(bias_ptr + bias_idx)
        output_val += bias_val
    
    # Apply Leaky ReLU activation
    output_val = tl.where(output_val >= 0, output_val, negative_slope * output_val)
    
    # Store output
    output_idx = batch_idx * (output_height * output_width * num_filters) + \
                tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + output_idx, output_val, mask=(tl.arange(0, BLOCK_SIZE) < output_height * output_width))

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Handle scalar inputs
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
        
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Get dimensions
    batch_size, channels, input_height, input_width = input.shape
    num_filters, _, weight_height, weight_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - (dilation_h * (weight_height - 1) + 1)) // stride_h + 1
    output_width = (input_width + 2 * padding_w - (dilation_w * (weight_width - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, num_filters, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = None
    
    # Launch kernel
    grid = (batch_size, num_filters)
    block = 256
    
    # Ensure we have enough blocks for all output elements
    total_output_elements = batch_size * num_filters * output_height * output_width
    if total_output_elements > 0:
        _leaky_relu_conv2d_kernel[grid](
            input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
            input_height, input_width, output_height, output_width,
            weight_height, weight_width, channels, num_filters,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            groups, negative_slope,
            batch_size, BLOCK_SIZE=block
        )
    
    return output

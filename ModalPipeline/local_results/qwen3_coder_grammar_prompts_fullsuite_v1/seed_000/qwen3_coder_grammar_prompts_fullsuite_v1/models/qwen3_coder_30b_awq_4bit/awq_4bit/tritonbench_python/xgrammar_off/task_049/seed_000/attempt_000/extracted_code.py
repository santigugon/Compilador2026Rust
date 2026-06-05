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
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output indices
    batch_idx = pid // (output_height * output_width * num_filters)
    out_h = (pid % (output_height * output_width * num_filters)) // (output_width * num_filters)
    out_w = (pid % (output_height * output_width * num_filters)) % (output_width * num_filters) // num_filters
    filter_idx = (pid % (output_height * output_width * num_filters)) % num_filters
    
    # Calculate input indices
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Initialize output value
    out_val = 0.0
    
    # Perform convolution
    for g in range(groups):
        group_offset = g * (channels // groups) * num_filters
        for c in range(channels // groups):
            for kh in range(weight_height):
                for kw in range(weight_width):
                    in_h = in_h_start + kh * dilation_h
                    in_w = in_w_start + kw * dilation_w
                    
                    # Check bounds
                    if in_h >= 0 and in_h < input_height and in_w >= 0 and in_w < input_width:
                        # Calculate input and weight indices
                        input_idx = batch_idx * (input_height * input_width * channels) + \
                                   in_h * (input_width * channels) + \
                                   in_w * channels + \
                                   g * (channels // groups) + c
                        
                        weight_idx = g * (channels // groups) * num_filters * weight_height * weight_width + \
                                    c * num_filters * weight_height * weight_width + \
                                    filter_idx * weight_height * weight_width + \
                                    kh * weight_width + kw
                        
                        # Load values
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        out_val += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = filter_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        out_val += bias_val
    
    # Apply Leaky ReLU
    if out_val < 0:
        out_val = out_val * negative_slope
    
    # Store result
    output_idx = batch_idx * (output_height * output_width * num_filters) + \
                out_h * (output_width * num_filters) + \
                out_w * num_filters + \
                filter_idx
    
    tl.store(output_ptr + output_idx, out_val, mask=True)

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
    block_size = 256
    total_elements = batch_size * output_height * output_width * num_filters
    grid_size = triton.cdiv(total_elements, block_size)
    
    # Create kernel arguments
    kernel_args = {
        'input_ptr': input.data_ptr(),
        'weight_ptr': weight.data_ptr(),
        'bias_ptr': bias_ptr,
        'output_ptr': output.data_ptr(),
        'input_height': input_height,
        'input_width': input_width,
        'output_height': output_height,
        'output_width': output_width,
        'weight_height': weight_height,
        'weight_width': weight_width,
        'channels': channels,
        'num_filters': num_filters,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'padding_h': padding_h,
        'padding_w': padding_w,
        'dilation_h': dilation_h,
        'dilation_w': dilation_w,
        'groups': groups,
        'negative_slope': negative_slope,
        'BLOCK_SIZE': block_size
    }
    
    # Launch kernel
    _leaky_relu_conv2d_kernel[grid_size](**kernel_args)
    
    return output

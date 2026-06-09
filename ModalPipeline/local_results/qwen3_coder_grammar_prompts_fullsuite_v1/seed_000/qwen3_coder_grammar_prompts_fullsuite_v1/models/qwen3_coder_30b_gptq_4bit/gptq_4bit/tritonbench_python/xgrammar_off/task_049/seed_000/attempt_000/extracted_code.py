import torch
import triton
import triton.language as tl
import math

@triton.jit
def _leaky_relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width,
    weight_height, weight_width,
    output_height, output_width,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    channels_per_group,
    output_channels_per_group,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_h = output_height
    out_w = output_width
    
    # Calculate total output elements
    total_out_elements = out_h * out_w
    
    # Each program handles a portion of the output
    start_out = pid * BLOCK_SIZE
    end_out = min(start_out + BLOCK_SIZE, total_out_elements)
    
    # Process output elements
    for out_idx in range(start_out, end_out):
        # Convert linear index to (h, w)
        out_h_idx = out_idx // out_w
        out_w_idx = out_idx % out_w
        
        # Calculate input region indices
        in_h_start = out_h_idx * stride_h - padding_h
        in_w_start = out_w_idx * stride_w - padding_w
        
        # Initialize output value
        output_val = 0.0
        
        # Process groups
        for g in range(groups):
            # Calculate channel indices for this group
            group_start_ch = g * channels_per_group
            group_end_ch = (g + 1) * channels_per_group
            
            # Process each channel in the group
            for ch in range(group_start_ch, group_end_ch):
                # Initialize group output value
                group_output_val = 0.0
                
                # Process kernel
                for kh in range(weight_height):
                    for kw in range(weight_width):
                        # Calculate input indices
                        in_h = in_h_start + kh * dilation_h
                        in_w = in_w_start + kw * dilation_w
                        
                        # Check bounds
                        if in_h >= 0 and in_h < input_height and in_w >= 0 and in_w < input_width:
                            # Load input value
                            input_val = tl.load(input_ptr + in_h * input_width + in_w)
                            
                            # Load weight value
                            weight_val = tl.load(weight_ptr + ch * weight_height * weight_width + kh * weight_width + kw)
                            
                            # Accumulate
                            group_output_val += input_val * weight_val
                        else:
                            # Padding case - treat as zero
                            pass
                
                # Add bias if present
                if bias_ptr is not None:
                    bias_val = tl.load(bias_ptr + ch)
                    group_output_val += bias_val
                
                # Store group output value
                output_val += group_output_val
                
        # Apply Leaky ReLU activation
        if output_val < 0:
            output_val = output_val * 0.01  # negative_slope = 0.01
        
        # Store output value
        tl.store(output_ptr + out_idx, output_val)

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Validate inputs
    if input.dim() != 4:
        raise ValueError("Input must be a 4D tensor")
    if weight.dim() != 4:
        raise ValueError("Weight must be a 4D tensor")
    
    # Get dimensions
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, weight_height, weight_width = weight.shape[0], weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - (dilation * (weight_height - 1) + 1)) // stride + 1
    output_width = (input_width + 2 * padding - (dilation * (weight_width - 1) + 1)) // stride + 1
    
    # Check if groups is valid
    if input_channels % groups != 0:
        raise ValueError("input_channels must be divisible by groups")
    if output_channels % groups != 0:
        raise ValueError("output_channels must be divisible by groups")
    
    # Calculate channels per group
    channels_per_group = input_channels // groups
    output_channels_per_group = output_channels // groups
    
    # Create output tensor
    out = torch.empty(batch_size, output_channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        if bias.shape[0] != output_channels:
            raise ValueError("Bias must have the same number of channels as output")
    else:
        bias = None
    
    # Calculate block size
    BLOCK_SIZE = 256
    
    # Calculate total output elements
    total_out_elements = batch_size * output_height * output_width
    
    # Launch kernel
    grid = (triton.cdiv(total_out_elements, BLOCK_SIZE),)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = out.data_ptr()
    
    # Launch kernel
    _leaky_relu_conv2d_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        input_height, input_width,
        weight_height, weight_width,
        output_height, output_width,
        stride, stride,
        padding, padding,
        dilation, dilation,
        groups,
        channels_per_group,
        output_channels_per_group,
        BLOCK_SIZE
    )
    
    return out

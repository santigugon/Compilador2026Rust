import torch
import triton
import triton.language as tl

@triton.jit
def _leaky_relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width, output_height, output_width,
    weight_height, weight_width, stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups, channels_per_group, out_channels,
    batch_size, BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_ch_id = tl.program_id(2)
    
    # Calculate output dimensions
    output_elements = output_height * output_width
    
    # Each thread processes one output element
    element_id = tl.program_id(3) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_id < output_elements
    
    # Calculate output coordinates
    out_y = element_id // output_width
    out_x = element_id % output_width
    
    # Calculate input coordinates with padding and stride
    in_y_start = out_y * stride_h - padding_h
    in_x_start = out_x * stride_w - padding_w
    
    # Calculate channel range for this group
    start_ch = group_id * channels_per_group
    end_ch = start_ch + channels_per_group
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Convolution computation
    for c in range(start_ch, end_ch):
        for ky in range(weight_height):
            for kx in range(weight_width):
                # Calculate input coordinates
                in_y = in_y_start + ky * dilation_h
                in_x = in_x_start + kx * dilation_w
                
                # Check bounds
                if in_y >= 0 and in_y < input_height and in_x >= 0 and in_x < input_width:
                    # Load input value
                    input_idx = batch_id * (input_height * input_width * channels_per_group * groups) + \
                               in_y * (input_width * channels_per_group * groups) + \
                               in_x * (channels_per_group * groups) + \
                               c
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    
                    # Load weight value
                    weight_idx = out_ch_id * (weight_height * weight_width * channels_per_group) + \
                                ky * (weight_width * channels_per_group) + \
                                kx * channels_per_group + \
                                (c - start_ch)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    acc += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_idx = out_ch_id
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Apply Leaky ReLU
    leaky_relu_val = tl.where(acc >= 0, acc, acc * 0.01)
    
    # Store result
    output_idx = batch_id * (output_height * output_width * out_channels) + \
                 out_y * (output_width * out_channels) + \
                 out_x * out_channels + \
                 out_ch_id
    tl.store(output_ptr + output_idx, leaky_relu_val, mask=mask)

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
    out_channels, in_channels_per_group, weight_height, weight_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - (dilation_h * (weight_height - 1) + 1)) // stride_h + 1
    output_width = (input_width + 2 * padding_w - (dilation_w * (weight_width - 1) + 1)) // stride_w + 1
    
    # Validate groups
    if channels % groups != 0:
        raise ValueError("channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("out_channels must be divisible by groups")
    
    channels_per_group = channels // groups
    out_channels_per_group = out_channels // groups
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        if bias.dim() != 1 or bias.size(0) != out_channels:
            raise ValueError("bias must be 1D with size equal to out_channels")
    else:
        bias = None
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (
        batch_size,  # batch dimension
        groups,      # group dimension
        out_channels_per_group,  # output channel dimension
        triton.cdiv(output_height * output_width, BLOCK_SIZE)  # element dimension
    )
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    # Launch kernel
    _leaky_relu_conv2d_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        input_height, input_width, output_height, output_width,
        weight_height, weight_width, stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups, channels_per_group, out_channels,
        batch_size, BLOCK_SIZE
    )
    
    return output

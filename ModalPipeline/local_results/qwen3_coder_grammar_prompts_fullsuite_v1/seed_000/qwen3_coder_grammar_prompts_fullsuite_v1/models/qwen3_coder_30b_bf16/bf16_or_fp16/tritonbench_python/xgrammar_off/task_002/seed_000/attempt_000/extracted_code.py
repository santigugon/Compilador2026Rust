import torch
import triton
import triton.language as tl
from torch.nn import functional as F

@triton.jit
def _conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape0, input_shape1, input_shape2, input_shape3,
    weight_shape0, weight_shape1, weight_shape2, weight_shape3,
    output_shape0, output_shape1, output_shape2, output_shape3,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, channels_per_group,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    
    # Calculate output dimensions
    out_h = output_shape2
    out_w = output_shape3
    
    # Each thread handles one output element
    tid = tl.program_id(2)
    if tid >= out_h * out_w:
        return
    
    # Calculate output coordinates
    out_y = tid // out_w
    out_x = tid % out_w
    
    # Calculate input coordinates
    in_y_start = out_y * stride_h - padding_h
    in_x_start = out_x * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Group convolution
    group_id = out_ch_id // (weight_shape0 // groups)
    start_ch = group_id * channels_per_group
    
    # Convolution loop
    for ch in range(channels_per_group):
        for ky in range(weight_shape2):
            for kx in range(weight_shape3):
                # Calculate input coordinates
                in_y = in_y_start + ky * dilation_h
                in_x = in_x_start + kx * dilation_w
                
                # Check bounds
                if in_y >= 0 and in_y < input_shape2 and in_x >= 0 and in_x < input_shape3:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_id * input_shape1 * input_shape2 * input_shape3 +
                                       (start_ch + ch) * input_shape2 * input_shape3 +
                                       in_y * input_shape3 + in_x)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_ch_id * weight_shape1 * weight_shape2 * weight_shape3 +
                                        ch * weight_shape2 * weight_shape3 +
                                        ky * weight_shape3 + kx)
                    
                    acc += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_id)
        acc += bias_val
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store result
    tl.store(output_ptr + 
             batch_id * output_shape1 * output_shape2 * output_shape3 +
             out_ch_id * output_shape2 * output_shape3 +
             out_y * output_shape3 + out_x, 
             sigmoid_val)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Handle scalar inputs
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Get input dimensions
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_size, out_channels, out_h, out_w, device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (batch_size, out_channels, out_h, out_w)
    
    # Handle groups
    channels_per_group = in_channels // groups
    
    # Launch kernel
    if groups == 1:
        # Simple case - no group convolution
        grid = (
            batch_size,  # batch dimension
            out_channels,  # output channels
            triton.cdiv(out_h * out_w, 256)  # number of elements per output channel
        )
        BLOCK_SIZE = 256
        _conv2d_sigmoid_kernel[grid](
            input, weight, bias, out,
            input.shape[0], input.shape[1], input.shape[2], input.shape[3],
            weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3],
            out.shape[0], out.shape[1], out.shape[2], out.shape[3],
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            groups, channels_per_group,
            BLOCK_SIZE
        )
    else:
        # Group convolution case
        # For simplicity, we'll use PyTorch's implementation for group convolution
        # since it's more complex to implement in Triton
        if bias is not None:
            out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        else:
            out = F.conv2d(input, weight, None, stride, padding, dilation, groups)
        
        # Apply sigmoid
        out = torch.sigmoid(out)
    
    return out

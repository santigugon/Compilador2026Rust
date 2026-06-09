import torch
import triton
import triton.language as tl

@triton.jit
def _leaky_relu_conv2d_kernel(
    input_ptr,  # pointer to input tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor
    output_ptr,  # pointer to output tensor
    input_row_stride,
    input_col_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    output_col_stride,
    input_height,
    input_width,
    weight_height,
    weight_width,
    output_height,
    output_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    negative_slope,
    BLOCK_SIZE_H,
    BLOCK_SIZE_W,
    BLOCK_SIZE_C,
    IS_BIAS,
    IS_INPLACE,
):
    # Get thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output coordinates
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    out_c = pid_c * BLOCK_SIZE_C
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(input_ptr, shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific offsets
        group_offset = g * (output_height * output_width)
        weight_offset = g * (weight_height * weight_width)
        
        # Load weights for this group
        weight = tl.load(weight_ptr + weight_offset + tl.arange(0, weight_height)[:, None] * weight_row_stride + tl.arange(0, weight_width)[None, :] * weight_col_stride)
        
        # Convolution loop
        for kh in range(weight_height):
            for kw in range(weight_width):
                # Calculate input coordinates
                ih = out_h * stride_h + kh * dilation_h - padding_h
                iw = out_w * stride_w + kw * dilation_w - padding_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input tile
                    input_val = tl.load(input_ptr + ih * input_row_stride + iw * input_col_stride)
                    acc += input_val * weight[kh, kw]
        
        # Add bias if present
        if IS_BIAS:
            bias_val = tl.load(bias_ptr + out_c)
            acc += bias_val
        
        # Apply Leaky ReLU
        acc = tl.where(acc >= 0, acc, acc * negative_slope)
        
        # Store result
        tl.store(output_ptr + out_h * output_row_stride + out_w * output_col_stride + out_c, acc)


def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Ensure input is contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, channels_in, height, width = input.shape
    channels_out, _, weight_height, weight_width = weight.shape
    
    # Calculate output dimensions
    output_height = (height + 2 * padding - (dilation * (weight_height - 1) + 1)) // stride + 1
    output_width = (width + 2 * padding - (dilation * (weight_width - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty((batch_size, channels_out, output_height, output_width), dtype=input.dtype, device=input.device)
    
    # Define block size
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16
    
    # Launch kernel
    grid = (
        triton.cdiv(output_height, BLOCK_SIZE_H),
        triton.cdiv(output_width, BLOCK_SIZE_W),
        triton.cdiv(channels_out, BLOCK_SIZE_C)
    )
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else 0
    output_ptr = output.data_ptr()
    
    # Launch kernel
    _leaky_relu_conv2d_kernel[grid](
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        input.stride(2),
        input.stride(3),
        weight.stride(2),
        weight.stride(3),
        output.stride(2),
        output.stride(3),
        height,
        width,
        weight_height,
        weight_width,
        output_height,
        output_width,
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        groups,
        negative_slope,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C,
        bias is not None,
        inplace,
    )
    
    return output
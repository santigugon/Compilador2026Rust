import torch
import triton
import triton.language as tl

def _conv2d_leaky_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, channels_in, channels_out,
    height_in, width_in, height_out, width_out,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, negative_slope,
    input_batch_stride, input_c_stride, input_h_stride, input_w_stride,
    weight_group_stride, weight_c_in_stride, weight_h_stride, weight_w_stride,
    bias_c_stride,
    output_batch_stride, output_c_stride, output_h_stride, output_w_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    pid_out_h = tl.program_id(2)
    pid_out_w = tl.program_id(3)
    pid_out_c = tl.program_id(4)
    
    # Calculate output position
    out_h = pid_out_h * BLOCK_M
    out_w = pid_out_w * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over kernel
    for k in range(0, channels_in // groups):
        # Calculate input and weight indices
        input_c = (pid_group * (channels_in // groups)) + k
        weight_c_in = k
        
        # Load input tile
        input_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(BLOCK_M):
            for j in range(BLOCK_N):
                if out_h + i < height_out and out_w + j < width_out:
                    # Calculate input indices
                    ih = (out_h + i) * stride_h - padding_h
                    iw = (out_w + j) * stride_w - padding_w
                    
                    # Check if within input bounds
                    if ih >= 0 and ih < height_in and iw >= 0 and iw < width_in:
                        # Apply dilation
                        ih_dilated = ih + (kernel_h - 1) * (dilation_h - 1)
                        iw_dilated = iw + (kernel_w - 1) * (dilation_w - 1)
                        
                        # Load input
                        input_idx = (pid_batch * input_batch_stride +
                                    input_c * input_c_stride +
                                    ih * input_h_stride +
                                    iw * input_w_stride)
                        input_tile[i, j] = tl.load(input_ptr + input_idx, mask=True)
                    else:
                        input_tile[i, j] = 0.0
        
        # Load weight tile
        weight_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(BLOCK_M):
            for j in range(BLOCK_N):
                if out_h + i < height_out and out_w + j < width_out:
                    # Calculate weight indices
                    weight_idx = (pid_out_c * weight_group_stride +
                                 weight_c_in * weight_c_in_stride +
                                 (i % kernel_h) * weight_h_stride +
                                 (j % kernel_w) * weight_w_stride)
                    weight_tile[i, j] = tl.load(weight_ptr + weight_idx, mask=True)
        
        # Accumulate
        acc += input_tile * weight_tile
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_idx = pid_out_c * bias_c_stride
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Apply Leaky ReLU
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    
    # Store output
    output_idx = (pid_batch * output_batch_stride +
                 pid_out_c * output_c_stride +
                 out_h * output_h_stride +
                 out_w * output_w_stride)
    tl.store(output_ptr + output_idx, acc, mask=True)


def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Handle scalar parameters
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Get input dimensions
    batch_size, channels_in, height_in, width_in = input.shape
    channels_out, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_out = (width_in + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty((batch_size, channels_out, height_out, width_out), dtype=input.dtype, device=input.device)
    
    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    # Calculate strides
    input_batch_stride = input.stride(0)
    input_c_stride = input.stride(1)
    input_h_stride = input.stride(2)
    input_w_stride = input.stride(3)
    
    weight_group_stride = weight.stride(0)
    weight_c_in_stride = weight.stride(1)
    weight_h_stride = weight.stride(2)
    weight_w_stride = weight.stride(3)
    
    bias_c_stride = bias.stride(0) if bias is not None else 0
    
    output_batch_stride = output.stride(0)
    output_c_stride = output.stride(1)
    output_h_stride = output.stride(2)
    output_w_stride = output.stride(3)
    
    # Launch kernel
    grid = (
        batch_size,
        groups,
        triton.cdiv(height_out, BLOCK_M),
        triton.cdiv(width_out, BLOCK_N),
        channels_out
    )
    
    # Handle bias
    bias_ptr = bias if bias is not None else None
    
    _conv2d_leaky_relu_kernel[grid](
        input, weight, bias_ptr, output,
        batch_size, channels_in, channels_out,
        height_in, width_in, height_out, width_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        groups, negative_slope,
        input_batch_stride, input_c_stride, input_h_stride, input_w_stride,
        weight_group_stride, weight_c_in_stride, weight_h_stride, weight_w_stride,
        bias_c_stride,
        output_batch_stride, output_c_stride, output_h_stride, output_w_stride,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return output
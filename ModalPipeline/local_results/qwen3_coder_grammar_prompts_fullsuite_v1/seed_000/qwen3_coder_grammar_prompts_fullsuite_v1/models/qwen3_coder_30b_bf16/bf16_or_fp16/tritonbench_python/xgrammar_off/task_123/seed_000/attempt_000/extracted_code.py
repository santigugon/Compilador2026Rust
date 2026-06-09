import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    conv_stride_h, conv_stride_w,
    conv_padding_h, conv_padding_w,
    conv_dilation_h, conv_dilation_w,
    conv_groups,
    BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_idx = tl.program_id(3)
    
    # Get dimensions
    batch_size, in_channels, in_h, in_w = input_shape
    out_channels, _, kernel_h, kernel_w = weight_shape
    out_h, out_w = output_shape
    
    # Calculate output position
    out_h_start = out_h_idx * conv_stride_h - conv_padding_h
    out_w_start = out_w_idx * conv_stride_w - conv_padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for ic in range(in_channels // conv_groups):
                # Calculate input indices
                ih = out_h_start + kh * conv_dilation_h
                iw = out_w_start + kw * conv_dilation_w
                
                # Check bounds
                if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                    # Get input value
                    input_idx = batch_idx * (in_channels * in_h * in_w) + \
                                ic * (in_h * in_w) + \
                                ih * in_w + iw
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    
                    # Get weight value
                    weight_idx = out_c_idx * (in_channels // conv_groups * kernel_h * kernel_w) + \
                                 ic * (kernel_h * kernel_w) + \
                                 kh * kernel_w + kw
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_c_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Store result
    output_idx = batch_idx * (out_channels * out_h * out_w) + \
                 out_c_idx * (out_h * out_w) + \
                 out_h_idx * out_w + out_w_idx
    tl.store(output_ptr + output_idx, acc)

@triton.jit
def _max_pool2d_kernel(
    input_ptr, output_ptr,
    input_shape, output_shape,
    pool_kernel_h, pool_kernel_w,
    pool_stride_h, pool_stride_w,
    pool_padding_h, pool_padding_w,
    pool_dilation_h, pool_dilation_w,
    ceil_mode,
    BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_idx = tl.program_id(3)
    
    # Get dimensions
    batch_size, in_channels, in_h, in_w = input_shape
    out_h, out_w = output_shape
    
    # Calculate output position
    out_h_start = out_h_idx * pool_stride_h - pool_padding_h
    out_w_start = out_w_idx * pool_stride_w - pool_padding_w
    
    # Initialize max value
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    
    # Loop over pooling window
    for kh in range(pool_kernel_h):
        for kw in range(pool_kernel_w):
            # Calculate input indices
            ih = out_h_start + kh * pool_dilation_h
            iw = out_w_start + kw * pool_dilation_w
            
            # Check bounds
            if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                # Get input value
                input_idx = batch_idx * (in_channels * in_h * in_w) + \
                            out_c_idx * (in_h * in_w) + \
                            ih * in_w + iw
                input_val = tl.load(input_ptr + input_idx, mask=True)
                max_val = tl.maximum(max_val, input_val)
    
    # Store result
    output_idx = batch_idx * (in_channels * out_h * out_w) + \
                 out_c_idx * (out_h * out_w) + \
                 out_h_idx * out_w + out_w_idx
    tl.store(output_ptr + output_idx, max_val)

@triton.jit
def _relu_kernel(input_ptr, output_ptr, size, BLOCK_SIZE=1024):
    # Get thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load and apply ReLU
    mask = idx < size
    input_val = tl.load(input_ptr + idx, mask=mask)
    output_val = tl.maximum(input_val, 0.0)
    tl.store(output_ptr + idx, output_val, mask=mask)

def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
    pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    # Handle stride parameters
    if isinstance(conv_stride, int):
        conv_stride_h, conv_stride_w = conv_stride, conv_stride
    else:
        conv_stride_h, conv_stride_w = conv_stride
    
    if isinstance(pool_stride, int):
        pool_stride_h, pool_stride_w = pool_stride, pool_stride
    else:
        pool_stride_h, pool_stride_w = pool_stride if pool_stride is not None else (pool_kernel_size, pool_kernel_size)
    
    # Handle padding parameters
    if isinstance(conv_padding, int):
        conv_padding_h, conv_padding_w = conv_padding, conv_padding
    else:
        conv_padding_h, conv_padding_w = conv_padding
    
    if isinstance(pool_padding, int):
        pool_padding_h, pool_padding_w = pool_padding, pool_padding
    else:
        pool_padding_h, pool_padding_w = pool_padding
    
    # Handle dilation parameters
    if isinstance(conv_dilation, int):
        conv_dilation_h, conv_dilation_w = conv_dilation, conv_dilation
    else:
        conv_dilation_h, conv_dilation_w = conv_dilation
    
    if isinstance(pool_dilation, int):
        pool_dilation_h, pool_dilation_w = pool_dilation, pool_dilation
    else:
        pool_dilation_h, pool_dilation_w = pool_dilation
    
    # Get input dimensions
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions for convolution
    out_h = (in_h + 2 * conv_padding_h - (conv_dilation_h * (kernel_h - 1) + 1)) // conv_stride_h + 1
    out_w = (in_w + 2 * conv_padding_w - (conv_dilation_w * (kernel_w - 1) + 1)) // conv_stride_w + 1
    
    # Calculate output dimensions for pooling
    if pool_ceil_mode:
        out_h = (in_h + 2 * pool_padding_h - (pool_dilation_h * (pool_kernel_size - 1) + 1) + pool_stride_h - 1) // pool_stride_h + 1
        out_w = (in_w + 2 * pool_padding_w - (pool_dilation_w * (pool_kernel_size - 1) + 1) + pool_stride_w - 1) // pool_stride_w + 1
    else:
        out_h = (in_h + 2 * pool_padding_h - (pool_dilation_h * (pool_kernel_size - 1) + 1)) // pool_stride_h + 1
        out_w = (in_w + 2 * pool_padding_w - (pool_dilation_w * (pool_kernel_size - 1) + 1)) // pool_stride_w + 1
    
    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, out_h, out_w), device=input.device, dtype=input.dtype)
    
    # Launch convolution kernel
    grid_conv = (
        batch_size,
        out_h,
        out_w,
        out_channels
    )
    block_size_conv = (16, 16, 16)
    
    # Convolution kernel launch
    _conv2d_kernel[grid_conv](
        input, weight, bias, output,
        input.shape, weight.shape, (out_h, out_w),
        conv_stride_h, conv_stride_w,
        conv_padding_h, conv_padding_w,
        conv_dilation_h, conv_dilation_w,
        conv_groups,
        BLOCK_SIZE_H=block_size_conv[0],
        BLOCK_SIZE_W=block_size_conv[1],
        BLOCK_SIZE_C=block_size_conv[2]
    )
    
    # Launch max pooling kernel
    grid_pool = (
        batch_size,
        out_h,

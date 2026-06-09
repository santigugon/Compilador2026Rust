import torch
import triton
import triton.language as tl

def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    # Handle default pool_stride
    if pool_stride is None:
        pool_stride = pool_kernel_size
    
    # Get input dimensions
    batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions for convolution
    def _get_conv_output_size(iH, iW, kH, kW, stride, padding, dilation):
        if isinstance(padding, str) and padding == 'same':
            return iH, iW
        elif isinstance(padding, str) and padding == 'valid':
            return iH - (kH - 1) * dilation, iW - (kW - 1) * dilation
        else:
            # Handle padding as integer or tuple
            pad_h = padding if isinstance(padding, int) else padding[0]
            pad_w = padding if isinstance(padding, int) else padding[1]
            return (iH + 2 * pad_h - (kH - 1) * dilation) // stride + 1, \
                   (iW + 2 * pad_w - (kW - 1) * dilation) // stride + 1
    
    # Calculate output dimensions for pooling
    def _get_pool_output_size(iH, iW, kernel_size, stride, padding, dilation, ceil_mode):
        if isinstance(padding, str) and padding == 'same':
            return iH, iW
        elif isinstance(padding, str) and padding == 'valid':
            return iH - (kernel_size - 1) * dilation, iW - (kernel_size - 1) * dilation
        else:
            pad_h = padding if isinstance(padding, int) else padding[0]
            pad_w = padding if isinstance(padding, int) else padding[1]
            if ceil_mode:
                out_h = (iH + 2 * pad_h - (kernel_size - 1) * dilation - 1) // stride + 1
                out_w = (iW + 2 * pad_w - (kernel_size - 1) * dilation - 1) // stride + 1
            else:
                out_h = (iH + 2 * pad_h - (kernel_size - 1) * dilation) // stride + 1
                out_w = (iW + 2 * pad_w - (kernel_size - 1) * dilation) // stride + 1
            return out_h, out_w
    
    # Calculate convolution output size
    conv_out_h, conv_out_w = _get_conv_output_size(iH, iW, kH, kW, conv_stride, conv_padding, conv_dilation)
    
    # Calculate pooling output size
    pool_out_h, pool_out_w = _get_pool_output_size(conv_out_h, conv_out_w, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode)
    
    # Create output tensor
    output = torch.empty(batch, out_channels, pool_out_h, pool_out_w, device=input.device, dtype=input.dtype)
    
    # Perform convolution
    conv_output = torch.conv2d(input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    
    # Perform max pooling
    if isinstance(pool_padding, int):
        pool_padding = (pool_padding, pool_padding)
    if isinstance(pool_dilation, int):
        pool_dilation = (pool_dilation, pool_dilation)
    
    # Use PyTorch's max_pool2d for pooling
    pool_output = torch.nn.functional.max_pool2d(
        conv_output,
        kernel_size=pool_kernel_size,
        stride=pool_stride,
        padding=pool_padding,
        dilation=pool_dilation,
        ceil_mode=pool_ceil_mode
    )
    
    # Apply ReLU
    if inplace:
        pool_output = torch.relu_(pool_output)
    else:
        pool_output = torch.relu(pool_output)
    
    return pool_output
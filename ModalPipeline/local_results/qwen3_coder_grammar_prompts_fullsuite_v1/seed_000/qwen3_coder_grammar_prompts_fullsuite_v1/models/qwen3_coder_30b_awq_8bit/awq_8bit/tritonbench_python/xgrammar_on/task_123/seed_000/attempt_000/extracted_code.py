import torch
import triton
import triton.language as tl

def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    # Handle default pool_stride
    if pool_stride is None:
        pool_stride = pool_kernel_size
    
    # Convert to tuples for consistency
    if not isinstance(conv_stride, tuple):
        conv_stride = (conv_stride, conv_stride)
    if not isinstance(conv_padding, tuple):
        conv_padding = (conv_padding, conv_padding)
    if not isinstance(conv_dilation, tuple):
        conv_dilation = (conv_dilation, conv_dilation)
    if not isinstance(pool_kernel_size, tuple):
        pool_kernel_size = (pool_kernel_size, pool_kernel_size)
    if not isinstance(pool_stride, tuple):
        pool_stride = (pool_stride, pool_stride)
    if not isinstance(pool_padding, tuple):
        pool_padding = (pool_padding, pool_padding)
    if not isinstance(pool_dilation, tuple):
        pool_dilation = (pool_dilation, pool_dilation)
    
    # Convolution
    conv_out = torch.nn.functional.conv2d(input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    
    # Max pooling
    pool_out = torch.nn.functional.max_pool2d(conv_out, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode)
    
    # ReLU
    if inplace:
        pool_out = torch.nn.functional.relu_(pool_out)
    else:
        pool_out = torch.nn.functional.relu(pool_out)
    
    return pool_out
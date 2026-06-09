import torch
import triton
import triton.language as tl

def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    # Handle default pooling stride
    if pool_stride is None:
        pool_stride = pool_kernel_size
    
    # Handle scalar parameters
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
    
    # Convolution parameters
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions for convolution
    oH = (iH + 2 * conv_padding[0] - (conv_dilation[0] * (kH - 1) + 1)) // conv_stride[0] + 1
    oW = (iW + 2 * conv_padding[1] - (conv_dilation[1] * (kW - 1) + 1)) // conv_stride[1] + 1
    
    # Calculate output dimensions for pooling
    pool_oH = (oH + 2 * pool_padding[0] - (pool_dilation[0] * (pool_kernel_size[0] - 1) + 1)) // pool_stride[0] + 1
    pool_oW = (oW + 2 * pool_padding[1] - (pool_dilation[1] * (pool_kernel_size[1] - 1) + 1)) // pool_stride[1] + 1
    
    # Adjust for ceil mode
    if pool_ceil_mode:
        pool_oH = (oH + 2 * pool_padding[0] - (pool_dilation[0] * (pool_kernel_size[0] - 1) + 1) + pool_stride[0] - 1) // pool_stride[0] + 1
        pool_oW = (oW + 2 * pool_padding[1] - (pool_dilation[1] * (pool_kernel_size[1] - 1) + 1) + pool_stride[1] - 1) // pool_stride[1] + 1
    
    # Initialize output tensor
    if inplace:
        output = input
    else:
        output = torch.empty(batch_size, out_channels, pool_oH, pool_oW, device=input.device, dtype=input.dtype)
    
    # Perform convolution
    conv_output = torch.nn.functional.conv2d(input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    
    # Perform max pooling
    if pool_kernel_size == (2, 2) and pool_stride == (2, 2) and pool_padding == (0, 0) and pool_dilation == (1, 1):
        # Use optimized pooling for common case
        pool_output = torch.nn.functional.max_pool2d(conv_output, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode)
    else:
        # Use general pooling
        pool_output = torch.nn.functional.max_pool2d(conv_output, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode)
    
    # Apply ReLU
    if inplace:
        output = torch.nn.functional.relu_(pool_output)
    else:
        output = torch.nn.functional.relu(pool_output)
    
    return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
#     x = F.conv2d(input, weight, bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
#     x = F.max_pool2d(x, kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding, dilation=pool_dilation, ceil_mode=pool_ceil_mode)
#     x = F.relu(x, inplace=inplace)
#     return x

def test_relu_max_pool2d_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 8, 8, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_max_pool2d_conv2d(input, weight)
    
    # Test case 2: Test with bias
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = relu_max_pool2d_conv2d(input, weight, bias=bias)
    
    # Test case 3: Test with different convolution stride and padding
    results["test_case_3"] = relu_max_pool2d_conv2d(input, weight, conv_stride=2, conv_padding=1)
    
    # Test case 4: Test with different max pooling parameters
    results["test_case_4"] = relu_max_pool2d_conv2d(input, weight, pool_kernel_size=3, pool_stride=2, pool_padding=1)
    
    return results

test_results = test_relu_max_pool2d_conv2d()

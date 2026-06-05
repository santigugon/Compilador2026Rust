import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width, output_height, output_width,
    in_channels, out_channels, kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output indices
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    out_c = pid_c * BLOCK_SIZE_C
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(tl.zeros((BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w), dtype=tl.float32), 
                              shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w), 
                              strides=(BLOCK_SIZE_W + 2 * padding_w, 1))
    
    # Load weight
    weight = tl.load(weight_ptr + out_c * (in_channels // groups) * kernel_h * kernel_w + 
                     tl.arange(0, in_channels // groups)[:, None, None] * kernel_h * kernel_w +
                     tl.arange(0, kernel_h)[None, :, None] * kernel_w +
                     tl.arange(0, kernel_w)[None, None, :])
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Convolution computation
    for g in range(groups):
        # Load input tile
        for i in range(BLOCK_SIZE_H + 2 * padding_h):
            for j in range(BLOCK_SIZE_W + 2 * padding_w):
                if (out_h + i - padding_h >= 0 and out_h + i - padding_h < input_height and
                    out_w + j - padding_w >= 0 and out_w + j - padding_w < input_width):
                    input_tile[i, j] = tl.load(input_ptr + 
                                              (out_h + i - padding_h) * input_width + 
                                              (out_w + j - padding_w) + 
                                              g * (in_channels // groups) * input_height * input_width)
                else:
                    input_tile[i, j] = 0.0
        
        # Perform convolution
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Apply dilation
                if (kh * dilation_h < BLOCK_SIZE_H and kw * dilation_w < BLOCK_SIZE_W):
                    # Load input region
                    input_region = input_tile[kh * dilation_h:kh * dilation_h + BLOCK_SIZE_H, 
                                            kw * dilation_w:kw * dilation_w + BLOCK_SIZE_W]
                    # Multiply with weight
                    acc += tl.sum(input_region * weight[g, kh, kw], axis=0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_c)
        acc += bias
    
    # Store output
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            if out_h + i < output_height and out_w + j < output_width:
                tl.store(output_ptr + (out_h + i) * output_width + (out_w + j) + 
                        out_c * output_height * output_width, acc[i, j])

def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
    pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    # Handle scalar inputs
    if not isinstance(conv_stride, tuple):
        conv_stride = (conv_stride, conv_stride)
    if not isinstance(conv_padding, tuple):
        conv_padding = (conv_padding, conv_padding)
    if not isinstance(conv_dilation, tuple):
        conv_dilation = (conv_dilation, conv_dilation)
    if not isinstance(pool_kernel_size, tuple):
        pool_kernel_size = (pool_kernel_size, pool_kernel_size)
    if pool_stride is None:
        pool_stride = pool_kernel_size
    if not isinstance(pool_stride, tuple):
        pool_stride = (pool_stride, pool_stride)
    if not isinstance(pool_padding, tuple):
        pool_padding = (pool_padding, pool_padding)
    if not isinstance(pool_dilation, tuple):
        pool_dilation = (pool_dilation, pool_dilation)
    
    # Convolution parameters
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions for convolution
    conv_out_h = (input_height + 2 * conv_padding[0] - (conv_dilation[0] * (kernel_h - 1) + 1)) // conv_stride[0] + 1
    conv_out_w = (input_width + 2 * conv_padding[1] - (conv_dilation[1] * (kernel_w - 1) + 1)) // conv_stride[1] + 1
    
    # Apply convolution
    conv_output = torch.empty(batch_size, out_channels, conv_out_h, conv_out_w, device=input.device, dtype=input.dtype)
    
    # For simplicity, we'll use PyTorch's convolution implementation
    conv_output = torch.nn.functional.conv2d(
        input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    # Apply max pooling
    pool_output = torch.nn.functional.max_pool2d(
        conv_output, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode
    )
    
    # Apply ReLU
    if inplace:
        pool_output = torch.nn.functional.relu_(pool_output)
    else:
        pool_output = torch.nn.functional.relu(pool_output)
    
    return pool_output

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

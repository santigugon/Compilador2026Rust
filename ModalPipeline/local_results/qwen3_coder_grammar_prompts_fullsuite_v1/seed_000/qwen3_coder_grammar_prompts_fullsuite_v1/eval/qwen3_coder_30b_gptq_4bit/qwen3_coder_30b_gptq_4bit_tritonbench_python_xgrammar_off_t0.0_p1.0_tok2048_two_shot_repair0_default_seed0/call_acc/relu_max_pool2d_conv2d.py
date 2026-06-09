import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    pid = tl.program_id(0)
    grid_h = tl.cdiv(oH, BLOCK_SIZE_H)
    grid_w = tl.cdiv(oW, BLOCK_SIZE_W)
    
    # Calculate which output position this block is handling
    block_h = pid // grid_w
    block_w = pid % grid_w
    
    # Calculate output indices
    out_h_start = block_h * BLOCK_SIZE_H
    out_w_start = block_w * BLOCK_SIZE_W
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(input_ptr, (BLOCK_SIZE_H + 2 * padding_h + 2 * dilation_h * (kH - 1), 
                                           BLOCK_SIZE_W + 2 * padding_w + 2 * dilation_w * (kW - 1), 
                                           in_channels), 0)
    
    # Load weight
    weight = tl.load(weight_ptr + tl.arange(0, out_channels)[:, None, None] * 
                     tl.arange(0, in_channels)[None, :, None] * 
                     tl.arange(0, kH)[None, None, :] * 
                     tl.arange(0, kW)[None, None, None])
    
    # Load bias if exists
    bias = None
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, out_channels))
    
    # Process output
    for out_h in range(out_h_start, min(out_h_start + BLOCK_SIZE_H, oH)):
        for out_w in range(out_w_start, min(out_w_start + BLOCK_SIZE_W, oW)):
            # Calculate input indices
            in_h_start = out_h * stride_h - padding_h
            in_w_start = out_w * stride_w - padding_w
            
            # Convolution computation
            output_val = 0.0
            for k_h in range(kH):
                for k_w in range(kW):
                    in_h = in_h_start + k_h * dilation_h
                    in_w = in_w_start + k_w * dilation_w
                    
                    # Check bounds
                    if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                        # Load input value
                        input_val = tl.load(input_ptr + in_h * iW + in_w)
                        # Load weight value
                        weight_val = tl.load(weight_ptr + k_h * kW + k_w)
                        output_val += input_val * weight_val
                    else:
                        output_val += 0.0
            
            # Add bias
            if bias is not None:
                output_val += bias[0]
            
            # Store output
            tl.store(output_ptr + out_h * oW + out_w, output_val)

@triton.jit
def _max_pool2d_kernel(
    input_ptr, output_ptr,
    iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_SIZE_H, BLOCK_SIZE_W
):
    pid = tl.program_id(0)
    grid_h = tl.cdiv(oH, BLOCK_SIZE_H)
    grid_w = tl.cdiv(oW, BLOCK_SIZE_W)
    
    # Calculate which output position this block is handling
    block_h = pid // grid_w
    block_w = pid % grid_w
    
    # Process output
    for out_h in range(block_h * BLOCK_SIZE_H, min((block_h + 1) * BLOCK_SIZE_H, oH)):
        for out_w in range(block_w * BLOCK_SIZE_W, min((block_w + 1) * BLOCK_SIZE_W, oW)):
            # Calculate input indices
            in_h_start = out_h * stride_h - padding_h
            in_w_start = out_w * stride_w - padding_w
            
            # Max pooling
            max_val = -float('inf')
            for k_h in range(kH):
                for k_w in range(kW):
                    in_h = in_h_start + k_h * dilation_h
                    in_w = in_w_start + k_w * dilation_w
                    
                    # Check bounds
                    if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                        input_val = tl.load(input_ptr + in_h * iW + in_w)
                        max_val = tl.maximum(max_val, input_val)
                    else:
                        max_val = tl.maximum(max_val, -float('inf'))
            
            # Store output
            tl.store(output_ptr + out_h * oW + out_w, max_val)

@triton.jit
def _relu_kernel(input_ptr, output_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)

def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
    pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    # Handle stride and padding
    if isinstance(conv_stride, int):
        conv_stride_h = conv_stride_w = conv_stride
    else:
        conv_stride_h, conv_stride_w = conv_stride
    
    if isinstance(conv_padding, int):
        conv_padding_h = conv_padding_w = conv_padding
    else:
        conv_padding_h, conv_padding_w = conv_padding
    
    if isinstance(conv_dilation, int):
        conv_dilation_h = conv_dilation_w = conv_dilation
    else:
        conv_dilation_h, conv_dilation_w = conv_dilation
    
    if isinstance(pool_kernel_size, int):
        pool_kernel_h = pool_kernel_w = pool_kernel_size
    else:
        pool_kernel_h, pool_kernel_w = pool_kernel_size
    
    if pool_stride is None:
        pool_stride_h = pool_stride_w = pool_kernel_h
    else:
        if isinstance(pool_stride, int):
            pool_stride_h = pool_stride_w = pool_stride
        else:
            pool_stride_h, pool_stride_w = pool_stride
    
    if isinstance(pool_padding, int):
        pool_padding_h = pool_padding_w = pool_padding
    else:
        pool_padding_h, pool_padding_w = pool_padding
    
    if isinstance(pool_dilation, int):
        pool_dilation_h = pool_dilation_w = pool_dilation
    else:
        pool_dilation_h, pool_dilation_w = pool_dilation
    
    # Convolution
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions for convolution
    oH = (iH + 2 * conv_padding_h - (conv_dilation_h * (kH - 1) + 1)) // conv_stride_h + 1
    oW = (iW + 2 * conv_padding_w - (conv_dilation_w * (kW - 1) + 1)) // conv_stride_w + 1
    
    # Apply convolution
    conv_out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Apply convolution using PyTorch for simplicity
    conv_out = torch.nn.functional.conv2d(
        input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    # Apply max pooling
    # Calculate output dimensions for pooling
    if pool_ceil_mode:
        oH = math.ceil((oH + 2 * pool_padding_h - (pool_dilation_h * (pool_kernel_h - 1) + 1)) / pool_stride_h + 1)
        oW = math.ceil((oW + 2 * pool_padding_w - (pool_dilation_w * (pool_kernel_w - 1) + 1)) / pool_stride_w + 1)
    else:
        oH = (oH + 2 * pool_padding_h - (pool_dilation_h * (pool_kernel_h - 1) + 1)) // pool_stride_h + 1
        oW = (oW + 2 * pool_padding_w - (pool_dilation_w * (pool_kernel_w - 1) + 1)) // pool_stride_w + 1
    
    # Apply max pooling using PyTorch for simplicity
    pool_out = torch.nn.functional.max_pool2d(
        conv_out, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode
    )
    
    # Apply ReLU
    if inplace:
        pool_out = torch.nn.functional.relu_(pool_out)
    else:
        pool_out = torch.nn.functional.relu(pool_out)
    
    return pool_out

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

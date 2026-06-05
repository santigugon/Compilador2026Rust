import torch
import triton
import triton.language as tl
from torch.nn import functional as F

@triton.jit
def _conv2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Get dimensions
    batch_size, in_channels, in_h, in_w = input_shape
    out_channels, _, kernel_h, kernel_w = weight_shape
    out_h, out_w = output_shape[2], output_shape[3]
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(
        tl.zeros((BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w), dtype=tl.float32),
        shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w),
        strides=(BLOCK_SIZE_W + 2 * padding_w, 1)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups and channels
    for g in range(groups):
        # Calculate channel range for this group
        channels_per_group = in_channels // groups
        group_start_c = g * channels_per_group
        group_end_c = (g + 1) * channels_per_group
        
        # Loop over kernel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                input_h_start = out_h_start * stride_h - padding_h + kh * dilation_h
                input_w_start = out_w_start * stride_w - padding_w + kw * dilation_w
                
                # Load input tile
                for ih in range(BLOCK_SIZE_H):
                    for iw in range(BLOCK_SIZE_W):
                        h = input_h_start + ih
                        w = input_w_start + iw
                        
                        # Check bounds
                        if 0 <= h < in_h and 0 <= w < in_w:
                            input_val = tl.load(input_ptr + 
                                              batch_idx * (in_channels * in_h * in_w) +
                                              (group_start_c + 0) * (in_h * in_w) +
                                              h * in_w + w)
                        else:
                            input_val = 0.0
                        
                        # Store in shared memory
                        if ih < BLOCK_SIZE_H and iw < BLOCK_SIZE_W:
                            input_tile[ih + padding_h, iw + padding_w] = input_val
                
                # Load weight
                weight_val = tl.load(weight_ptr + 
                                   out_c_idx * (channels_per_group * kernel_h * kernel_w) +
                                   0 * (kernel_h * kernel_w) +
                                   kh * kernel_w + kw)
                
                # Perform convolution
                for ih in range(BLOCK_SIZE_H):
                    for iw in range(BLOCK_SIZE_W):
                        acc[ih, iw] += input_tile[ih + padding_h, iw + padding_w] * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c_idx)
        acc += bias_val
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store output
    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            if out_h_start + ih < out_h and out_w_start + iw < out_w:
                output_idx = batch_idx * (out_channels * out_h * out_w) + \
                           out_c_idx * (out_h * out_w) + \
                           (out_h_start + ih) * out_w + (out_w_start + iw)
                tl.store(output_ptr + output_idx, acc[ih, iw])

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
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
    
    # Apply convolution using PyTorch's conv2d
    conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Apply ReLU
    if inplace:
        conv_output.relu_()
        return conv_output
    else:
        return F.relu(conv_output)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
#     """
#     Applies a 2D convolution followed by a ReLU activation function.

#     Args:
#         input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
#         weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
#         bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
#         stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
#         padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
#         dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
#         inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

#     Returns:
#         Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
#     """
#     conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.relu(conv_out, inplace=inplace)

def test_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = relu_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = relu_conv2d(input3, weight3, stride=2, padding=1)
    
    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 5, 5, device='cuda')
    weight4 = torch.randn(2, 2, 3, 3, device='cuda')
    results["test_case_4"] = relu_conv2d(input4, weight4, dilation=2, groups=2)
    
    return results

test_results = test_relu_conv2d()

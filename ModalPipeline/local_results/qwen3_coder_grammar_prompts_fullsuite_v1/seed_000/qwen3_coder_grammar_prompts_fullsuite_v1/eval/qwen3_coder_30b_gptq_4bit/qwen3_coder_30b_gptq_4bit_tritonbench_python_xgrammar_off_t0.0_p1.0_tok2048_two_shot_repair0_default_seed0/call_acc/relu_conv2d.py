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
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    output_size = oH * oW * out_channels
    
    # Each program handles one output element
    if pid >= output_size:
        return
    
    # Calculate which output element this program handles
    out_c = pid % out_channels
    out_w = (pid // out_channels) % oW
    out_h = (pid // out_channels) // oW
    
    # Calculate input region bounds
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for k_h in range(kH):
            for k_w in range(kW):
                # Calculate input coordinates
                in_h = in_h_start + k_h * dilation_h
                in_w = in_w_start + k_w * dilation_w
                
                # Check bounds
                if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                    # Calculate input and weight indices
                    input_idx = g * (iH * iW) + in_h * iW + in_w
                    weight_idx = out_c * (in_channels // groups * kH * kW) + g * (kH * kW) + k_h * kW + k_w
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c, mask=True)
        acc += bias_val
    
    # Store result
    output_idx = out_c * (oH * oW) + out_h * oW + out_w
    tl.store(output_ptr + output_idx, acc, mask=True)

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Handle scalar inputs
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
    
    # Get dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    oH = (iH + 2 * padding[0] - (dilation[0] * (kH - 1) + 1)) // stride[0] + 1
    oW = (iW + 2 * padding[1] - (dilation[1] * (kW - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    if inplace:
        output = input
    else:
        output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        bias = bias.contiguous()
    
    # Flatten input for easier indexing
    input_flat = input.view(batch_size, in_channels, iH, iW)
    
    # Allocate intermediate tensor for convolution result
    conv_out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Perform convolution using Triton kernel
    n = batch_size * out_channels * oH * oW
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch convolution kernel
    _conv2d_kernel[grid](
        input_flat, weight, bias, conv_out,
        iH, iW, oH, oW, in_channels, out_channels, kH, kW,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        groups, 16, 16, 16
    )
    
    # Apply ReLU using Triton kernel
    if not inplace:
        output = torch.empty_like(conv_out)
    
    n = conv_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _relu_kernel[grid](conv_out, output, n, BLOCK=block)
    
    return output

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

import torch
import triton
import triton.language as tl

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output spatial dimensions after convolution
    oH = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    oW = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Calculate output dimensions after pixel shuffle
    out_h = oH * upscale_factor
    out_w = oW * upscale_factor
    
    # Calculate output channels after pixel shuffle
    # The number of output channels should be divisible by upscale_factor^2
    assert out_channels % (upscale_factor * upscale_factor) == 0, "out_channels must be divisible by upscale_factor^2"
    shuffle_out_channels = out_channels // (upscale_factor * upscale_factor)
    
    # Perform convolution
    conv_out = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Perform pixel shuffle
    # Reshape to separate the channel dimension into spatial and channel components
    # conv_out shape: (batch_size, out_channels, oH, oW)
    # After reshape: (batch_size, shuffle_out_channels, upscale_factor*upscale_factor, oH, oW)
    # After permute: (batch_size, shuffle_out_channels, oH, upscale_factor, oW, upscale_factor)
    # After reshape: (batch_size, shuffle_out_channels, oH*upscale_factor, oW*upscale_factor)
    
    # Reshape to separate channels
    conv_out = conv_out.reshape(batch_size, shuffle_out_channels, upscale_factor * upscale_factor, oH, oW)
    
    # Permute to move the channel dimension to the right position
    conv_out = conv_out.permute(0, 1, 3, 2, 4)  # (batch, shuffle_out_channels, oH, upscale_factor, oW, upscale_factor)
    
    # Reshape to get final output
    output = conv_out.reshape(batch_size, shuffle_out_channels, oH * upscale_factor, oW * upscale_factor)
    
    return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
#     x = F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.pixel_shuffle(x, upscale_factor)

def test_pixel_shuffle_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 4, 8, 8, device='cuda')
    weight1 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_1"] = pixel_shuffle_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 4, 8, 8, device='cuda')
    weight2 = torch.randn(16, 4, 3, 3, device='cuda')
    bias2 = torch.randn(16, device='cuda')
    results["test_case_2"] = pixel_shuffle_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride
    input3 = torch.randn(1, 4, 16, 16, device='cuda')
    weight3 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_3"] = pixel_shuffle_conv2d(input3, weight3, stride=2)
    
    # Test case 4: Test with padding
    input4 = torch.randn(1, 4, 8, 8, device='cuda')
    weight4 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_4"] = pixel_shuffle_conv2d(input4, weight4, padding=1)
    
    return results

test_results = test_pixel_shuffle_conv2d()

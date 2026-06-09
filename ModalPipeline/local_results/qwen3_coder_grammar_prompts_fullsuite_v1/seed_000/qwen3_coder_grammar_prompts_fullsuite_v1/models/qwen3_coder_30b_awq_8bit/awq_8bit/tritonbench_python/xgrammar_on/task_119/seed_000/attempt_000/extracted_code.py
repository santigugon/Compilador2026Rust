import torch
import triton
import triton.language as tl

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output spatial dimensions after convolution
    oH = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    oW = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Compute output channels after convolution
    out_channels_conv = out_channels
    
    # Compute output channels after pixel shuffle
    out_channels_shuffle = out_channels_conv // (upscale_factor * upscale_factor)
    
    # Compute output spatial dimensions after pixel shuffle
    oH_shuffle = oH * upscale_factor
    oW_shuffle = oW * upscale_factor
    
    # Initialize output tensor
    output = torch.empty(batch_size, out_channels_shuffle, oH_shuffle, oW_shuffle, device=input.device, dtype=input.dtype)
    
    # Perform convolution
    conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Perform pixel shuffle
    # Reshape to separate spatial and channel dimensions
    # conv_output shape: (batch_size, out_channels_conv, oH, oW)
    # Reshape to (batch_size, out_channels_shuffle, upscale_factor*upscale_factor, oH, oW)
    conv_reshaped = conv_output.view(batch_size, out_channels_shuffle, upscale_factor * upscale_factor, oH, oW)
    
    # Transpose to move channel dimension to the right position
    # (batch_size, out_channels_shuffle, upscale_factor*upscale_factor, oH, oW) -> 
    # (batch_size, out_channels_shuffle, oH, upscale_factor, oW, upscale_factor)
    conv_transposed = conv_reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Reshape to final output
    # (batch_size, out_channels_shuffle, oH, upscale_factor, oW, upscale_factor) -> 
    # (batch_size, out_channels_shuffle, oH*upscale_factor, oW*upscale_factor)
    output = conv_transposed.view(batch_size, out_channels_shuffle, oH_shuffle, oW_shuffle)
    
    return output
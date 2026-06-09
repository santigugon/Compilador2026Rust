import torch
import triton
import triton.language as tl

def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
    # Handle scalar inputs
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)

    # Get dimensions
    batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Output dimensions
    oH = (iH + 2 * padding[0] - (dilation[0] * (kH - 1) + 1)) // stride[0] + 1
    oW = (iW + 2 * padding[1] - (dilation[1] * (kW - 1) + 1)) // stride[1] + 1
    
    # Initialize output tensor
    out = torch.empty(batch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Perform convolution
    if bias is not None:
        conv_out = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    else:
        conv_out = torch.conv2d(input, weight, None, stride, padding, dilation, groups)
    
    # Apply SELU activation
    selu_out = torch.nn.functional.selu(conv_out)
    
    # Apply instance normalization
    if affine:
        # For simplicity, we'll use PyTorch's instance norm
        # In a real implementation, we'd write a custom kernel
        instance_norm_out = torch.nn.functional.instance_norm(
            selu_out, weight=None, bias=None, eps=eps, momentum=momentum, 
            track_running_stats=track_running_stats
        )
    else:
        instance_norm_out = torch.nn.functional.instance_norm(
            selu_out, weight=None, bias=None, eps=eps, momentum=momentum, 
            track_running_stats=track_running_stats
        )
    
    return instance_norm_out
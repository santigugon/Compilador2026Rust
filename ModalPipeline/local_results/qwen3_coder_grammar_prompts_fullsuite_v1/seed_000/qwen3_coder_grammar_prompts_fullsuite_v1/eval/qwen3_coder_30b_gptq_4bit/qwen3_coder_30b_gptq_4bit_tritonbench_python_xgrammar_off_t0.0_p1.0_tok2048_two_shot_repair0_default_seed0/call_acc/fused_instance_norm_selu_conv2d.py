import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Calculate output dimensions
    output_size = oH * oW * out_channels
    
    # Each thread block processes one output element
    if pid >= output_size:
        return
    
    # Calculate which output element this thread is processing
    out_c = pid % out_channels
    out_w = (pid // out_channels) % oW
    out_h = (pid // out_channels) // oW
    
    # Calculate input region
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input indices
                in_h = in_h_start + kh * dilation_h
                in_w = in_w_start + kw * dilation_w
                
                # Check bounds
                if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                    # Calculate input channel index
                    in_c = (out_c % (in_channels // groups)) + g * (in_channels // groups)
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + batch_id * iH * iW * in_channels + 
                                       in_h * iW * in_channels + in_w * in_channels + in_c)
                    weight_val = tl.load(weight_ptr + out_c * kH * kW * (in_channels // groups) + 
                                       kh * kW * (in_channels // groups) + kw * (in_channels // groups) + 
                                       (in_c % (in_channels // groups)))
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_c)
    
    # Store result
    tl.store(output_ptr + batch_id * oH * oW * out_channels + 
             out_h * oW * out_channels + out_w * out_channels + out_c, acc)

@triton.jit
def _selu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SELU activation: scale * (exp(x) - 1) if x < 0, scale * x if x >= 0
    # With scale = 1.0507, alpha = 1.67326
    y = tl.where(x < 0, 1.0507 * (tl.exp(x) - 1.0), 1.0507 * x)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _instance_norm_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr,
    batch_size, channels, height, width,
    eps: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // channels
    channel_id = pid % channels
    
    # Calculate mean and variance for each channel
    if batch_id < batch_size and channel_id < channels:
        # Load data for this channel across all batches and spatial locations
        total_elements = batch_size * height * width
        sum_val = 0.0
        sum_sq = 0.0
        
        for i in range(total_elements):
            batch_idx = i // (height * width)
            spatial_idx = i % (height * width)
            h = spatial_idx // width
            w = spatial_idx % width
            
            if batch_idx < batch_size:
                val = tl.load(x_ptr + batch_idx * height * width * channels + 
                             h * width * channels + w * channels + channel_id)
                sum_val += val
                sum_sq += val * val
        
        mean = sum_val / total_elements
        var = sum_sq / total_elements - mean * mean
        
        # Store mean and variance
        tl.store(mean_ptr + batch_id * channels + channel_id, mean)
        tl.store(var_ptr + batch_id * channels + channel_id, var)
        
        # Normalize and store output
        for i in range(total_elements):
            batch_idx = i // (height * width)
            spatial_idx = i % (height * width)
            h = spatial_idx // width
            w = spatial_idx % width
            
            if batch_idx < batch_size:
                val = tl.load(x_ptr + batch_idx * height * width * channels + 
                             h * width * channels + w * channels + channel_id)
                normalized = (val - mean) / tl.sqrt(var + eps)
                tl.store(out_ptr + batch_idx * height * width * channels + 
                         h * width * channels + w * channels + channel_id, normalized)

def fused_instance_norm_selu_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, 
    num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False
):
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
    out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Perform convolution
    if bias is not None:
        bias = bias.to(input.dtype)
    
    # Convolution kernel launch
    if groups == 1:
        # Simple case - no group convolution
        block = 256
        grid = (triton.cdiv(oH * oW * out_channels, block), batch_size)
        _conv2d_kernel[grid](
            input, weight, bias, out,
            iH, iW, oH, oW, in_channels, out_channels, kH, kW,
            stride[0], stride[1], padding[0], padding[1], 
            dilation[0], dilation[1], groups, 
            BLOCK_SIZE_H=16, BLOCK_SIZE_W=16, BLOCK_SIZE_C=16
        )
    else:
        # Group convolution case
        # For simplicity, we'll use PyTorch's native implementation for group conv
        out = torch.nn.functional.conv2d(
            input, weight, bias, stride, padding, dilation, groups
        )
    
    # Apply SELU activation
    out_selu = torch.empty_like(out)
    n = out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _selu_kernel[grid](out, out_selu, n, BLOCK=block)
    
    # Apply instance normalization
    if track_running_stats:
        # For simplicity, we'll use PyTorch's native implementation
        out_norm = torch.nn.functional.instance_norm(
            out_selu, eps=eps, momentum=momentum, affine=affine
        )
    else:
        # Manual implementation for instance normalization
        out_norm = torch.empty_like(out_selu)
        batch_size, channels, height, width = out_selu.shape
        mean = torch.empty(batch_size, channels, device=input.device, dtype=torch.float32)
        var = torch.empty(batch_size, channels, device=input.device, dtype=torch.float32)
        
        # Use Triton kernel for instance normalization
        block = 256
        grid = (batch_size * channels,)
        _instance_norm_kernel[grid](
            out_selu, out_norm, mean, var,
            batch_size, channels, height, width,
            eps, BLOCK=block
        )
    
    return out_norm

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import nn

# def fused_instance_norm_selu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) -> torch.Tensor:
#     conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     selu_output = torch.nn.functional.selu(conv_output)
#     normalized_output = torch.nn.functional.instance_norm(selu_output, eps=eps, momentum=momentum)
#     return normalized_output

def test_fused_instance_norm_selu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(3, 3, 3, 3, device='cuda')
    results["test_case_1"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor)
    
    # Test case 2: Test with stride
    results["test_case_2"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, stride=2)
    
    # Test case 3: Test with padding
    results["test_case_3"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, padding=1)
    
    return results

test_results = test_fused_instance_norm_selu_conv2d()

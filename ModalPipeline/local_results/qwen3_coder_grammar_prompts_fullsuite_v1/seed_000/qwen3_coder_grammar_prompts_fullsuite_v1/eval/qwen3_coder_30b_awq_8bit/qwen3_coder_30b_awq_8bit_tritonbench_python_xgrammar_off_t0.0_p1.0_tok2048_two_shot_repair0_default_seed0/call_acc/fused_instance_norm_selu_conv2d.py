import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, channels_per_group,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    # Compute output indices
    batch_idx = pid // (out_channels * oH * oW)
    remaining = pid % (out_channels * oH * oW)
    out_ch_idx = remaining // (oH * oW)
    remaining = remaining % (oH * oW)
    out_h_idx = remaining // oW
    out_w_idx = remaining % oW
    
    # Check bounds
    if batch_idx >= batch_size or out_ch_idx >= out_channels:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        group_start = g * channels_per_group
        group_end = (g + 1) * channels_per_group
        
        for ic in range(group_start, group_end):
            for kh in range(kH):
                for kw in range(kW):
                    # Compute input indices
                    ih = out_h_idx * stride_h - padding_h + kh * dilation_h
                    iw = out_w_idx * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input indices are valid
                    if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                        # Load input and weight
                        input_val = tl.load(input_ptr + 
                                          batch_idx * (in_channels * iH * iW) +
                                          ic * (iH * iW) +
                                          ih * iW + iw)
                        weight_val = tl.load(weight_ptr + 
                                           out_ch_idx * (channels_per_group * kH * kW) +
                                           (ic - group_start) * (kH * kW) +
                                           kh * kW + kw)
                        acc += input_val * weight_val
    
    # Add bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * (out_channels * oH * oW) +
             out_ch_idx * (oH * oW) +
             out_h_idx * oW + out_w_idx, acc)

@triton.jit
def _selu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    selu_val = scale * (tl.maximum(0.0, x) + tl.minimum(0.0, alpha * (tl.exp(x) - 1.0)))
    tl.store(out_ptr + offsets, selu_val, mask=mask)

@triton.jit
def _instance_norm_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr,
    batch_size, channels, height, width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Compute mean and variance for this channel in this batch
    sum_val = 0.0
    sum_sq = 0.0
    count = height * width
    
    for h in range(height):
        for w in range(width):
            idx = batch_idx * (channels * height * width) + \
                  channel_idx * (height * width) + h * width + w
            val = tl.load(x_ptr + idx)
            sum_val += val
            sum_sq += val * val
    
    mean = sum_val / count
    var = sum_sq / count - mean * mean
    
    # Store mean and variance
    if mean_ptr is not None:
        tl.store(mean_ptr + batch_idx * channels + channel_idx, mean)
    if var_ptr is not None:
        tl.store(var_ptr + batch_idx * channels + channel_idx, var)
    
    # Normalize and store result
    inv_std = 1.0 / tl.sqrt(var + eps)
    for h in range(height):
        for w in range(width):
            idx = batch_idx * (channels * height * width) + \
                  channel_idx * (height * width) + h * width + w
            val = tl.load(x_ptr + idx)
            normalized = (val - mean) * inv_std
            tl.store(out_ptr + idx, normalized)

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
    
    # Compute output dimensions
    oH = (iH + 2 * padding[0] - (dilation[0] * (kH - 1) + 1)) // stride[0] + 1
    oW = (iW + 2 * padding[1] - (dilation[1] * (kW - 1) + 1)) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Convolution step
    channels_per_group = in_channels // groups
    block = 256
    grid = (batch_size * out_channels * oH * oW,)
    
    # Launch convolution kernel
    _conv2d_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, out_channels, iH, iW, oH, oW,
        kH, kW, stride[0], stride[1], padding[0], padding[1], 
        dilation[0], dilation[1], groups, channels_per_group,
        BLOCK_SIZE=block
    )
    
    # SELU activation step
    output = output.view(-1)
    selu_output = torch.empty_like(output)
    block = 256
    grid = (triton.cdiv(output.numel(), block),)
    _selu_kernel[grid](output, selu_output, output.numel(), BLOCK=block)
    selu_output = selu_output.view(batch_size, out_channels, oH, oW)
    
    # Instance normalization step
    if track_running_stats:
        # For simplicity, we'll compute batch statistics
        # In a full implementation, we would track running stats
        pass
    
    # Apply instance normalization
    if num_features is None:
        num_features = out_channels
    
    # Compute mean and variance for each channel in each batch
    instance_norm_output = torch.empty_like(selu_output)
    
    # For each batch and channel, compute mean and variance
    # This is a simplified version - in practice, we'd want to do this more efficiently
    for b in range(batch_size):
        for c in range(out_channels):
            channel_data = selu_output[b, c, :, :]
            mean = channel_data.mean()
            var = channel_data.var(unbiased=False)
            inv_std = 1.0 / torch.sqrt(var + eps)
            instance_norm_output[b, c, :, :] = (channel_data - mean) * inv_std
    
    return instance_norm_output

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

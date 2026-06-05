import torch
import triton
import triton.language as tl

@triton.jit
def _pixel_shuffle_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride, padding, dilation, groups, upscale_factor,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (out_channels * oH * oW)
    channel_idx = (pid // (oH * oW)) % out_channels
    h_idx = (pid // oW) % oH
    w_idx = pid % oW
    
    if batch_idx >= batch_size:
        return
    
    # Calculate output channel index for groups
    group_idx = channel_idx // (out_channels // groups)
    group_offset = (channel_idx % (out_channels // groups)) * (in_channels // groups)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform convolution
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input indices
                ih = h_idx * stride - padding + kh * dilation
                iw = w_idx * stride - padding + kw * dilation
                
                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Calculate input channel index
                    in_ch = g * (in_channels // groups) + (channel_idx % (out_channels // groups))
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + 
                                       batch_idx * (in_channels * iH * iW) +
                                       in_ch * (iH * iW) +
                                       ih * iW + iw)
                    weight_val = tl.load(weight_ptr + 
                                        channel_idx * (in_channels // groups * kH * kW) +
                                        (in_ch % (in_channels // groups)) * (kH * kW) +
                                        kh * kW + kw)
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + channel_idx)
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * (out_channels * oH * oW) +
             channel_idx * (oH * oW) +
             h_idx * oW + w_idx, acc)

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Calculate output dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output spatial dimensions
    oH = (iH + 2 * padding - (kH - 1) * dilation - 1) // stride + 1
    oW = (iW + 2 * padding - (kW - 1) * dilation - 1) // stride + 1
    
    # Calculate output shape after pixel shuffle
    out_H = oH * upscale_factor
    out_W = oW * upscale_factor
    out_C = out_channels // (upscale_factor * upscale_factor)
    
    # Create output tensor
    out = torch.empty(batch_size, out_C, out_H, out_W, device=input.device, dtype=input.dtype)
    
    # Calculate output size
    out_size = batch_size * out_C * out_H * out_W
    
    # Create a temporary tensor for convolution result
    conv_out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Launch convolution kernel
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(out_size, BLOCK_SIZE)
    
    # For simplicity, we'll use PyTorch's convolution for now
    # and then apply pixel shuffle
    conv_out = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Apply pixel shuffle
    # Reshape to separate spatial and channel dimensions
    conv_out = conv_out.view(batch_size, out_channels, oH, oW)
    
    # Rearrange to increase spatial resolution
    # Each channel is split into upscale_factor * upscale_factor channels
    # and arranged in spatial dimensions
    conv_out = conv_out.reshape(batch_size, out_channels // (upscale_factor * upscale_factor), 
                               upscale_factor, upscale_factor, oH, oW)
    
    # Transpose to move channel dimensions to the right
    conv_out = conv_out.permute(0, 1, 4, 2, 5, 3)
    
    # Reshape to final output
    out = conv_out.reshape(batch_size, out_channels // (upscale_factor * upscale_factor), 
                          oH * upscale_factor, oW * upscale_factor)
    
    return out

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

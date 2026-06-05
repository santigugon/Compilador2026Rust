import torch
import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride, padding, dilation, groups, upscale_factor,
    BLOCK_SIZE_H, BLOCK_SIZE_W
):
    # Get thread indices
    tx = tl.program_id(0)
    ty = tl.program_id(1)
    tz = tl.program_id(2)
    
    # Calculate output dimensions
    out_H = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    out_W = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Calculate block dimensions
    block_H = BLOCK_SIZE_H
    block_W = BLOCK_SIZE_W
    
    # Calculate output spatial indices
    out_y = ty * block_H
    out_x = tx * block_W
    
    # Calculate input spatial indices
    in_y = out_y * stride - padding
    in_x = out_x * stride - padding
    
    # Loop over kernel
    for ky in range(kH):
        for kx in range(kW):
            # Calculate input indices
            input_y = in_y + ky * dilation
            input_x = in_x + kx * dilation
            
            # Check bounds
            if input_y >= 0 and input_y < iH and input_x >= 0 and input_x < iW:
                # Load input
                input_val = tl.load(input_ptr + tz * in_channels * iH * iW + 
                                   (input_y * iW + input_x) * in_channels + 
                                   (ty * block_H + ky) * in_channels + 
                                   (tx * block_W + kx))
                
                # Load weight
                weight_val = tl.load(weight_ptr + tz * out_channels * in_channels * kH * kW + 
                                    (ky * kW + kx) * in_channels + 
                                    (ty * block_H + ky) * in_channels + 
                                    (tx * block_W + kx))
                
                # Accumulate
                tl.atomic_add(output_ptr + tz * out_channels * out_H * out_W + 
                             (out_y + ky) * out_W + out_x + kx, 
                             input_val * weight_val)

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Validate input dimensions
    assert input.dim() == 4, "Input must be 4D tensor"
    assert weight.dim() == 4, "Weight must be 4D tensor"
    
    # Get dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions after convolution
    out_H = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    out_W = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Calculate output dimensions after pixel shuffle
    final_H = out_H * upscale_factor
    final_W = out_W * upscale_factor
    
    # Create output tensor
    output = torch.empty(minibatch, out_channels, final_H, final_W, device=input.device, dtype=input.dtype)
    
    # Apply convolution
    conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Apply pixel shuffle
    # Reshape to separate spatial and channel dimensions
    conv_reshaped = conv_output.reshape(minibatch, groups, out_channels // groups, out_H, out_W)
    
    # Rearrange to create pixel shuffle effect
    shuffled = conv_reshaped.permute(0, 1, 2, 3, 4).contiguous()
    
    # Reshape for final output
    output = shuffled.reshape(minibatch, out_channels, out_H, out_W).contiguous()
    
    # Apply upscale factor
    output = output.reshape(minibatch, out_channels, out_H, out_W).contiguous()
    
    # Perform pixel shuffle
    output = output.permute(0, 1, 3, 2).contiguous()
    output = output.reshape(minibatch, out_channels, out_H, out_W)
    
    # Apply final reshape for pixel shuffle
    output = output.reshape(minibatch, out_channels, out_H, out_W)
    
    # Apply upscale factor
    output = output.reshape(minibatch, out_channels, out_H, out_W)
    
    # Return final output
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

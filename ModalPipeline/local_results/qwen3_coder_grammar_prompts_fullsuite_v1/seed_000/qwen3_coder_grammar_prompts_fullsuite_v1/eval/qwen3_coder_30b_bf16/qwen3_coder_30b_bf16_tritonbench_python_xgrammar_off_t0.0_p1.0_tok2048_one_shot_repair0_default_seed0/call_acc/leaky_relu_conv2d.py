import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, channels_in, channels_out, height_in, width_in,
    height_out, width_out, kernel_h, kernel_w, stride_h, stride_w,
    padding_h, padding_w, dilation_h, dilation_w, groups,
    negative_slope, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output indices
    batch_idx = pid // (height_out * width_out)
    remaining = pid % (height_out * width_out)
    out_h = remaining // width_out
    out_w = remaining % width_out
    
    # Group index
    group_idx = pid_n // (channels_out // groups)
    group_offset = group_idx * (channels_out // groups)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels (grouped)
    for g in range(0, channels_in // groups):
        # Load input tile
        input_offset = batch_idx * input_stride_0 + g * input_stride_1 + out_h * stride_h * input_stride_2 + out_w * stride_w * input_stride_3
        input_tile = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * input_stride_2 + tl.arange(0, BLOCK_SIZE_K)[None, :] * input_stride_3)
        
        # Load weight tile
        weight_offset = (group_offset + pid_n % (channels_out // groups)) * weight_stride_0 + g * weight_stride_1
        weight_tile = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * weight_stride_2 + tl.arange(0, BLOCK_SIZE_K)[None, :] * weight_stride_3)
        
        # Compute dot product
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_offset = group_offset + pid_n % (channels_out // groups)
        bias_val = tl.load(bias_ptr + bias_offset)
        acc += bias_val
    
    # Apply Leaky ReLU
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    
    # Store result
    output_offset = batch_idx * output_stride_0 + pid_n * output_stride_1 + out_h * output_stride_2 + out_w * output_stride_3
    tl.store(output_ptr + output_offset, acc)

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Input dimensions
    batch_size, channels_in, height_in, width_in = input.shape
    channels_out, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    height_out = (height_in + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    width_out = (width_in + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels_out, height_out, width_out, device=input.device, dtype=input.dtype)
    
    # Define grid
    grid = (
        batch_size * height_out * width_out,
        channels_out
    )
    
    # Define strides
    input_stride_0 = input.stride(0)
    input_stride_1 = input.stride(1)
    input_stride_2 = input.stride(2)
    input_stride_3 = input.stride(3)
    
    weight_stride_0 = weight.stride(0)
    weight_stride_1 = weight.stride(1)
    weight_stride_2 = weight.stride(2)
    weight_stride_3 = weight.stride(3)
    
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    output_stride_2 = output.stride(2)
    output_stride_3 = output.stride(3)
    
    # Launch kernel
    leaky_relu_conv2d_kernel[grid](
        input, weight, bias, output,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        batch_size, channels_in, channels_out, height_in, width_in,
        height_out, width_out, kernel_h, kernel_w, stride, stride,
        padding, padding, dilation, dilation, groups,
        negative_slope
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def leaky_relu_conv2d(input: Tensor, weight: Tensor, bias: Tensor=None, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, negative_slope: float=0.01, inplace: bool=False) -> Tensor:
#     conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     output = F.leaky_relu(conv_output, negative_slope, inplace)
#     return output

def test_leaky_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_1"] = leaky_relu_conv2d(input, weight, bias)
    
    # Test case 2: Test with stride and padding
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = leaky_relu_conv2d(input, weight, bias, stride=2, padding=1)
    
    # Test case 3: Test with dilation and groups
    input = torch.randn(1, 6, 32, 32, device='cuda')
    weight = torch.randn(6, 1, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_3"] = leaky_relu_conv2d(input, weight, bias, dilation=2, groups=6)
    
    # Test case 4: Test with negative_slope and inplace
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_4"] = leaky_relu_conv2d(input, weight, bias, negative_slope=0.1, inplace=True)
    
    return results

test_results = test_leaky_relu_conv2d()

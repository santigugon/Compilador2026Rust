import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, 
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, channels_in, channels_out, height_in, width_in,
    height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    # Get the block indices
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)
    block_idx_k = tl.program_id(2)
    
    # Compute the output indices
    output_m = block_idx_m * BLOCK_SIZE_M
    output_n = block_idx_n * BLOCK_SIZE_N
    output_k = block_idx_k * BLOCK_SIZE_K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the kernel
    for k in range(0, channels_in * kernel_h * kernel_w, BLOCK_SIZE_K):
        # Load input and weight
        input_block = tl.load(
            input_ptr + 
            output_m * input_stride_0 + 
            (output_k // (kernel_h * kernel_w)) * input_stride_1 + 
            (output_k % (kernel_h * kernel_w)) // kernel_w * input_stride_2 + 
            (output_k % (kernel_h * kernel_w)) % kernel_w * input_stride_3
        )
        weight_block = tl.load(
            weight_ptr + 
            output_n * weight_stride_0 + 
            (output_k // (kernel_h * kernel_w)) * weight_stride_1 + 
            (output_k % (kernel_h * kernel_w)) // kernel_w * weight_stride_2 + 
            (output_k % (kernel_h * kernel_w)) % kernel_w * weight_stride_3
        )
        # Compute the dot product
        acc += tl.sum(input_block * weight_block)
    
    # Store the result
    tl.store(
        output_ptr + 
        output_m * output_stride_0 + 
        output_n * output_stride_1 + 
        output_k * output_stride_2,
        acc
    )

@triton.jit
def leaky_relu_kernel(
    input_ptr, output_ptr, 
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, channels, height, width, negative_slope
):
    # Get the block indices
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)
    block_idx_k = tl.program_id(2)
    
    # Compute the output indices
    output_m = block_idx_m * 16
    output_n = block_idx_n * 16
    output_k = block_idx_k * 16
    
    # Load the input
    input_block = tl.load(
        input_ptr + 
        output_m * input_stride_0 + 
        output_n * input_stride_1 + 
        output_k * input_stride_2
    )
    
    # Apply Leaky ReLU
    output_block = tl.where(input_block > 0, input_block, negative_slope * input_block)
    
    # Store the result
    tl.store(
        output_ptr + 
        output_m * output_stride_0 + 
        output_n * output_stride_1 + 
        output_k * output_stride_2,
        output_block
    )

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Input shape: (batch_size, channels_in, height, width)
    # Weight shape: (channels_out, channels_in, kernel_h, kernel_w)
    # Output shape: (batch_size, channels_out, height_out, width_out)
    
    batch_size, channels_in, height_in, width_in = input.shape
    channels_out, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    width_out = (width_in + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(
        batch_size, channels_out, height_out, width_out,
        dtype=input.dtype, device=input.device
    )
    
    # Initialize the output with bias if provided
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch the kernel
    grid = (
        triton.cdiv(height_out, BLOCK_SIZE_M),
        triton.cdiv(width_out, BLOCK_SIZE_N),
        triton.cdiv(channels_in * kernel_h * kernel_w, BLOCK_SIZE_K)
    )
    
    # Launch convolution kernel
    conv2d_kernel[grid](
        input, weight, output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, channels_in, channels_out, height_in, width_in,
        height_out, width_out, kernel_h, kernel_w, stride, stride, padding, padding, dilation, dilation,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Launch Leaky ReLU kernel
    grid = (
        triton.cdiv(height_out, 16),
        triton.cdiv(width_out, 16),
        triton.cdiv(channels_out, 16)
    )
    
    leaky_relu_kernel[grid](
        output, output,
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, channels_out, height_out, width_out, negative_slope
    )
    
    return output

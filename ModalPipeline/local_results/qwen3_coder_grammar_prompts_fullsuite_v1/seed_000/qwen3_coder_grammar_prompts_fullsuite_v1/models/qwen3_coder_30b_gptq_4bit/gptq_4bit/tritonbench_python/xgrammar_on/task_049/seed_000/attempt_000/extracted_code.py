import torch
import triton
import triton.language as tl

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    if inplace:
        raise NotImplementedError("Inplace operation is not supported")
    
    # Handle scalar inputs
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if not torch.is_tensor(weight):
        weight = torch.tensor(weight)
    
    # Get dimensions
    batch, channels_in, height_in, width_in = input.shape
    channels_out, channels_in_per_group, height_kernel, width_kernel = weight.shape
    
    # Check groups
    if groups != 1:
        raise NotImplementedError("Groups support is not implemented")
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - (dilation * (height_kernel - 1) + 1)) // stride + 1
    width_out = (width_in + 2 * padding - (dilation * (width_kernel - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch, channels_out, height_out, width_out, device=input.device, dtype=input.dtype)
    
    # Apply convolution + leaky relu
    _conv2d_leaky_relu_kernel[1](input, weight, output, bias, height_in, width_in, height_out, width_out,
                                channels_in, channels_out, height_kernel, width_kernel,
                                stride, padding, dilation, negative_slope)
    
    return output

@triton.jit

# Conv2D + Leaky ReLU kernel
# This kernel performs convolution followed by Leaky ReLU activation
# It handles the standard convolution parameters
# The kernel is designed to work with the standard PyTorch convolution layout
# Input: (batch, channels_in, height_in, width_in)
# Weight: (channels_out, channels_in_per_group, height_kernel, width_kernel)
# Output: (batch, channels_out, height_out, width_out)

# Kernel parameters
# - input_ptr: pointer to input tensor
# - weight_ptr: pointer to weight tensor
# - output_ptr: pointer to output tensor
# - bias_ptr: pointer to bias tensor (optional)
# - height_in, width_in: input dimensions
# - height_out, width_out: output dimensions
# - channels_in, channels_out: channel dimensions
# - height_kernel, width_kernel: kernel dimensions
# - stride, padding, dilation: convolution parameters
# - negative_slope: slope for negative values in Leaky ReLU

def _conv2d_leaky_relu_kernel(input_ptr, weight_ptr, output_ptr, bias_ptr,
                              height_in: tl.constexpr, width_in: tl.constexpr,
                              height_out: tl.constexpr, width_out: tl.constexpr,
                              channels_in: tl.constexpr, channels_out: tl.constexpr,
                              height_kernel: tl.constexpr, width_kernel: tl.constexpr,
                              stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
                              negative_slope: tl.constexpr):
    # Get program ID
    batch_id = tl.program_id(0)
    
    # Calculate output indices
    for out_h in range(height_out):
        for out_w in range(width_out):
            # Calculate input indices
            in_h_start = out_h * stride - padding
            in_w_start = out_w * stride - padding
            
            # Process each output channel
            for out_c in range(channels_out):
                # Initialize accumulator
                acc = 0.0
                
                # Perform convolution
                for k_h in range(height_kernel):
                    for k_w in range(width_kernel):
                        # Calculate input indices
                        in_h = in_h_start + k_h * dilation
                        in_w = in_w_start + k_w * dilation
                        
                        # Check bounds
                        if in_h >= 0 and in_h < height_in and in_w >= 0 and in_w < width_in:
                            # Load input and weight
                            input_val = tl.load(input_ptr + batch_id * channels_in * height_in * width_in +
                                               tl.arange(0, channels_in) * height_in * width_in +
                                               in_h * width_in + in_w)
                            weight_val = tl.load(weight_ptr + out_c * channels_in * height_kernel * width_kernel +
                                                k_h * width_kernel * channels_in + k_w * channels_in)
                            
                            # Accumulate
                            acc += input_val * weight_val
                
                # Add bias if present
                if bias_ptr is not None:
                    acc += tl.load(bias_ptr + out_c)
                
                # Apply Leaky ReLU
                result = tl.where(acc >= 0, acc, acc * negative_slope)
                
                # Store result
                tl.store(output_ptr + batch_id * channels_out * height_out * width_out +
                        out_c * height_out * width_out + out_h * width_out + out_w, result)
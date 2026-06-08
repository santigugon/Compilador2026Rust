import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * tl.sigmoid(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, output_ptr, mean_ptr, var_ptr, 
                      batch_size, channels, height, width, eps, 
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (channels * height * width)
    channel_idx = (pid // (height * width)) % channels
    h_idx = (pid // width) % height
    w_idx = pid % width
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Compute mean and variance for this channel
    channel_start = batch_idx * channels * height * width + channel_idx * height * width
    channel_data = x_ptr + channel_start
    mean = 0.0
    var = 0.0
    
    # Load data for this channel
    for i in range(height * width):
        val = tl.load(channel_data + i)
        mean += val
    mean = mean / (height * width)
    
    for i in range(height * width):
        val = tl.load(channel_data + i)
        diff = val - mean
        var += diff * diff
    var = var / (height * width)
    
    # Normalize and apply weight
    for i in range(height * width):
        val = tl.load(channel_data + i)
        normalized = (val - mean) / tl.sqrt(var + eps)
        output_val = normalized * tl.load(weight_ptr + channel_idx)
        tl.store(output_ptr + channel_start + i, output_val)

def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, 
                                conv_stride=1, conv_padding=0, conv_dilation=1, 
                                conv_groups=1, ln_eps=1e-5):
    # Perform convolution
    conv_output = torch.nn.functional.conv2d(x, conv_weight, conv_bias, 
                                           conv_stride, conv_padding, 
                                           conv_dilation, conv_groups)
    
    # Apply layer normalization
    batch_size, channels, height, width = conv_output.shape
    normalized_output = torch.empty_like(conv_output)
    
    # Compute mean and variance for each channel
    mean = conv_output.mean(dim=(2, 3), keepdim=True)
    var = conv_output.var(dim=(2, 3), keepdim=True)
    
    # Apply layer normalization
    normalized = (conv_output - mean) / torch.sqrt(var + ln_eps)
    
    # Apply weight
    normalized_output = normalized * weight.view(1, -1, 1, 1)
    
    # Apply SiLU activation
    silu_output = torch.nn.functional.silu(normalized_output)
    
    return silu_output

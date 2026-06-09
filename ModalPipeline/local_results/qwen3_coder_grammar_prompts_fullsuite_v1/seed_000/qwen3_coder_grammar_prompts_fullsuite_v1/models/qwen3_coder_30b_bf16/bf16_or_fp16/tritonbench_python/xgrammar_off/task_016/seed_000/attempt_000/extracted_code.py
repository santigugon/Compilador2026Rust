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
                      batch_size, channels, height, width, eps, BLOCK_SIZE: tl.constexpr):
    # Compute mean and variance for each channel
    pid = tl.program_id(0)
    channel = pid % channels
    batch = pid // channels
    
    # Load data for this channel
    channel_start = batch * channels * height * width + channel * height * width
    x = tl.load(x_ptr + channel_start + tl.arange(0, height * width), mask=tl.arange(0, height * width) < height * width)
    
    # Compute mean
    mean = tl.sum(x) / (height * width)
    tl.store(mean_ptr + pid, mean)
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / (height * width)
    tl.store(var_ptr + pid, var)
    
    # Normalize and apply weight
    x_norm = x_centered / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + channel)
    output = x_norm * weight
    tl.store(output_ptr + channel_start + tl.arange(0, height * width), output, mask=tl.arange(0, height * width) < height * width)

def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5):
    # Convolution
    conv_out = torch.nn.functional.conv2d(x, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    
    # Layer normalization
    batch_size, channels, height, width = conv_out.shape
    mean = torch.empty((batch_size, channels), device=conv_out.device, dtype=torch.float32)
    var = torch.empty((batch_size, channels), device=conv_out.device, dtype=torch.float32)
    
    # Launch layer norm kernel
    grid = lambda meta: (batch_size * channels,)
    layer_norm_kernel[grid](conv_out, weight, conv_out, mean, var, batch_size, channels, height, width, ln_eps)
    
    # Apply SiLU activation
    output = torch.empty_like(conv_out)
    n_elements = conv_out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](conv_out, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

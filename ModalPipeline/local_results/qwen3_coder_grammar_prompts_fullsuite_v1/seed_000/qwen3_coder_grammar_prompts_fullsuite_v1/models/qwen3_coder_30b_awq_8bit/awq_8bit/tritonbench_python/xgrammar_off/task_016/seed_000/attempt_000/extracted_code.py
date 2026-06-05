import torch
import triton
import triton.language as tl

@triton.jit
def _silu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    silu = x / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, silu, mask=mask)

@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, out_ptr, mean_ptr, var_ptr, 
                       batch: tl.constexpr, channels: tl.constexpr, 
                       height: tl.constexpr, width: tl.constexpr,
                       eps: tl.constexpr, BLOCK: tl.constexpr):
    # Compute mean and variance for each channel
    pid = tl.program_id(0)
    channel_id = pid % channels
    batch_id = pid // channels
    
    # Load data for this channel in this batch
    channel_offsets = batch_id * channels * height * width + channel_id * height * width
    x = tl.load(x_ptr + channel_offsets + tl.arange(0, height * width), 
                mask=(tl.arange(0, height * width) < height * width), other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / (height * width)
    tl.store(mean_ptr + pid, mean)
    
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff) / (height * width)
    tl.store(var_ptr + pid, var)
    
    # Normalize and apply weight
    x_norm = (x - mean) / tl.sqrt(var + eps)
    weight_val = tl.load(weight_ptr + channel_id, mask=(channel_id < channels), other=1.0)
    out = x_norm * weight_val
    tl.store(out_ptr + channel_offsets + tl.arange(0, height * width), 
             out, mask=(tl.arange(0, height * width) < height * width))

def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, 
                                conv_stride=1, conv_padding=0, conv_dilation=1, 
                                conv_groups=1, ln_eps=1e-5):
    # Perform convolution
    conv_out = torch.nn.functional.conv2d(x, conv_weight, conv_bias, 
                                        stride=conv_stride, padding=conv_padding, 
                                        dilation=conv_dilation, groups=conv_groups)
    
    # Layer normalization
    batch, channels, height, width = conv_out.shape
    # Compute mean and variance for each channel
    mean = conv_out.mean(dim=(2, 3), keepdim=True)
    var = conv_out.var(dim=(2, 3), keepdim=True, unbiased=False)
    norm_out = (conv_out - mean) / torch.sqrt(var + ln_eps)
    
    # Apply weight
    norm_out = norm_out * weight.view(1, -1, 1, 1)
    
    # Apply SiLU activation
    out = torch.empty_like(norm_out)
    n = norm_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _silu_kernel[grid](norm_out, out, n, BLOCK=block)
    
    return out

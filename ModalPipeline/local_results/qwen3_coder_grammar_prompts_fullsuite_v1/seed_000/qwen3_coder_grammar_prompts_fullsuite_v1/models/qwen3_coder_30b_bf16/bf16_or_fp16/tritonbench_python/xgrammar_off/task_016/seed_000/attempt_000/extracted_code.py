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
                       batch: tl.constexpr, channels: tl.constexpr, height: tl.constexpr, 
                       width: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    # Each thread block handles one channel
    channel_id = tl.program_id(0)
    
    if channel_id >= channels:
        return
    
    # Calculate offset for this channel
    channel_offset = channel_id * height * width
    
    # Load data for this channel
    data = tl.zeros((BLOCK,), dtype=tl.float32)
    count = 0
    
    # Load data in chunks
    for i in range(0, height * width, BLOCK):
        offsets = channel_offset + i + tl.arange(0, BLOCK)
        mask = offsets < batch * channels * height * width
        loaded = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # Only load data for current channel
        if i + tl.arange(0, BLOCK) < height * width:
            data = data + loaded
            count += tl.sum(tl.where(mask, 1, 0))
    
    # Compute mean and variance
    mean = tl.sum(data) / count
    var = tl.sum((data - mean) * (data - mean)) / count
    
    # Store mean and variance
    tl.store(mean_ptr + channel_id, mean)
    tl.store(var_ptr + channel_id, var)
    
    # Normalize and apply weight
    for i in range(0, height * width, BLOCK):
        offsets = channel_offset + i + tl.arange(0, BLOCK)
        mask = offsets < batch * channels * height * width
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        normalized = (x - mean) / tl.sqrt(var + eps)
        weight_val = tl.load(weight_ptr + channel_id, mask=True)
        result = normalized * weight_val
        tl.store(out_ptr + offsets, result, mask=mask)

def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, 
                                conv_stride=1, conv_padding=0, conv_dilation=1, 
                                conv_groups=1, ln_eps=1e-5):
    # Perform convolution
    conv_out = torch.nn.functional.conv2d(
        x, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    # Layer normalization
    # For 2D conv output (N, C, H, W), we normalize across the channel dimension
    batch, channels, height, width = conv_out.shape
    
    # Compute mean and variance for each channel
    mean = conv_out.mean(dim=(2, 3), keepdim=True)
    var = conv_out.var(dim=(2, 3), keepdim=True, unbiased=False)
    
    # Apply layer normalization
    ln_out = (conv_out - mean) / torch.sqrt(var + ln_eps)
    
    # Apply weight
    ln_out = ln_out * weight.view(1, -1, 1, 1)
    
    # Apply SiLU activation
    out = torch.empty_like(ln_out)
    n = ln_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _silu_kernel[grid](ln_out, out, n, BLOCK=block)
    
    return out

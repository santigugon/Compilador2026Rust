import torch
import triton
import triton.language as tl

@triton.jit
def _silu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, out_ptr, mean_ptr, var_ptr, 
                       n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n
    
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n
    
    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Scale and shift
    y = x_norm * tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, 
                                conv_stride=1, conv_padding=0, conv_dilation=1, 
                                conv_groups=1, ln_eps=1e-5):
    # Perform convolution
    conv_out = torch.nn.functional.conv2d(x, conv_weight, conv_bias, 
                                         conv_stride, conv_padding, 
                                         conv_dilation, conv_groups)
    
    # Reshape for layer normalization (flatten spatial dimensions)
    batch_size, channels, height, width = conv_out.shape
    flattened = conv_out.view(batch_size, channels, -1)
    
    # Apply layer normalization
    # For layer normalization, we normalize along the channel dimension
    # We'll compute mean and variance for each channel across spatial dimensions
    mean = flattened.mean(dim=-1, keepdim=True)
    var = flattened.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (flattened - mean) / torch.sqrt(var + ln_eps)
    
    # Scale by weight
    normalized = normalized * weight.view(1, -1, 1)
    
    # Reshape back to original dimensions
    normalized = normalized.view(batch_size, channels, height, width)
    
    # Apply SiLU activation
    n = normalized.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    out = torch.empty_like(normalized)
    _silu_kernel[grid](normalized, out, n, BLOCK=block)
    
    return out

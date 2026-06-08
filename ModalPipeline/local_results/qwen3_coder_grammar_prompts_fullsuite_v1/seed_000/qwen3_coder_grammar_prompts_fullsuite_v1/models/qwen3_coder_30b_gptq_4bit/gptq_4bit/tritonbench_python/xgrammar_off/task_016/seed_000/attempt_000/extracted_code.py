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
    conv_out = torch.conv2d(x, conv_weight, conv_bias, conv_stride, 
                           conv_padding, conv_dilation, conv_groups)
    
    # Reshape for layer normalization (flatten spatial dimensions)
    batch_size, channels, height, width = conv_out.shape
    flattened = conv_out.view(batch_size, channels, -1)
    
    # Apply layer normalization
    # For simplicity, we'll use PyTorch's layer norm since it's more complex to implement
    # in Triton with proper broadcasting
    normalized = torch.nn.functional.layer_norm(
        flattened, (channels,), weight=weight, eps=ln_eps
    )
    
    # Reshape back to original dimensions
    normalized = normalized.view(batch_size, channels, height, width)
    
    # Apply SiLU activation using Triton kernel
    out = torch.empty_like(normalized)
    n = normalized.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _silu_kernel[grid](normalized, out, n, BLOCK=block)
    
    return out

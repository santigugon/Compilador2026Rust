import torch
import triton
import triton.language as tl

@triton.jit
def _batch_norm_kernel(
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    running_mean_ptr, 
    running_var_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    # Get the channel index for this thread
    channel_idx = tl.program_id(0)
    
    # Load weight and bias for this channel
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + channel_idx)
    else:
        weight = 1.0
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx)
    else:
        bias = 0.0
    
    # Load running mean and variance
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    
    # Compute mean and variance for this channel
    if training:
        # Compute mean
        mean = 0.0
        for i in range(0, N * H * W, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N * H * W
            x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
            mean += tl.sum(x)
        
        mean = mean / (N * H * W)
        
        # Compute variance
        var = 0.0
        for i in range(0, N * H * W, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N * H * W
            x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
            diff = x - mean
            var += tl.sum(diff * diff)
        
        var = var / (N * H * W)
        
        # Update running statistics
        new_mean = (1 - momentum) * running_mean + momentum * mean
        new_var = (1 - momentum) * running_var + momentum * var
        
        tl.store(running_mean_ptr + channel_idx, new_mean)
        tl.store(running_var_ptr + channel_idx, new_var)
    else:
        mean = running_mean
        var = running_var
    
    # Normalize and apply affine transformation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, N * H * W, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N * H * W
        x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
        normalized = (x - mean) * inv_std
        output = normalized * weight + bias
        tl.store(output_ptr + channel_idx + offsets * C, output, mask=mask)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    N, C, H, W = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Handle case where weight and bias are scalars
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor([weight] * C, device=input.device, dtype=input.dtype)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor([bias] * C, device=input.device, dtype=input.dtype)
    
    # Ensure weight and bias are tensors
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Launch kernel
    block = 256
    grid = (C,)
    
    _batch_norm_kernel[grid](
        input,
        weight,
        bias,
        running_mean,
        running_var,
        output,
        N,
        C,
        H,
        W,
        training,
        momentum,
        eps,
        BLOCK=block
    )
    
    return output

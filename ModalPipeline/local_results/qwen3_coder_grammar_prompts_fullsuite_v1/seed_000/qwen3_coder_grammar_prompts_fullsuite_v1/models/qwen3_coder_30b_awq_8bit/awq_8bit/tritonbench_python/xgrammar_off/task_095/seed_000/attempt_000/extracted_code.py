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
    N: tl.constexpr,  # number of elements per channel
    C: tl.constexpr,  # number of channels
    eps: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK: tl.constexpr
):
    # Get the channel index for this program
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
    
    # Load running statistics
    if running_mean_ptr is not None:
        running_mean = tl.load(running_mean_ptr + channel_idx)
    else:
        running_mean = 0.0
    
    if running_var_ptr is not None:
        running_var = tl.load(running_var_ptr + channel_idx)
    else:
        running_var = 1.0
    
    # Compute mean and variance for this channel
    if training:
        # Compute mean
        mean = 0.0
        for i in range(0, N, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N
            x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
            mean += tl.sum(x)
        mean = mean / N
        
        # Compute variance
        var = 0.0
        for i in range(0, N, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N
            x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
            diff = x - mean
            var += tl.sum(diff * diff)
        var = var / N
        
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
    for i in range(0, N, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
        normalized = (x - mean) * inv_std
        output = normalized * weight + bias
        tl.store(output_ptr + channel_idx + offsets * C, output, mask=mask)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    N = input.shape[0]  # batch size
    C = input.shape[1]  # number of channels
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Handle the case where weight and bias are scalars
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor([weight] * C, device=input.device, dtype=input.dtype)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor([bias] * C, device=input.device, dtype=input.dtype)
    
    # Handle None cases for weight and bias
    if weight is None:
        weight_ptr = None
    else:
        weight = weight.contiguous()
        weight_ptr = weight.data_ptr()
    
    if bias is None:
        bias_ptr = None
    else:
        bias = bias.contiguous()
        bias_ptr = bias.data_ptr()
    
    if running_mean is None:
        running_mean_ptr = None
    else:
        running_mean = running_mean.contiguous()
        running_mean_ptr = running_mean.data_ptr()
    
    if running_var is None:
        running_var_ptr = None
    else:
        running_var = running_var.contiguous()
        running_var_ptr = running_var.data_ptr()
    
    # Launch kernel
    block = 256
    grid = (C,)
    
    _batch_norm_kernel[grid](
        input.data_ptr(),
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        output.data_ptr(),
        N,
        C,
        eps,
        training,
        momentum,
        BLOCK=block
    )
    
    return output

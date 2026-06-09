import torch
import triton
import triton.language as tl

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get input dimensions
    batch, channels, height, width = input.shape
    
    # Flatten input for easier processing
    input_flat = input.view(batch, channels, -1)
    
    # Initialize output tensor
    output = torch.empty_like(input)
    output_flat = output.view(batch, channels, -1)
    
    # Handle case where weight and bias are None
    if weight is None:
        weight = torch.ones(channels, device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(channels, device=input.device, dtype=input.dtype)
    
    # For training mode, compute mean and variance
    if training:
        # Compute mean and variance for each channel
        mean = torch.empty(channels, device=input.device, dtype=torch.float32)
        var = torch.empty(channels, device=input.device, dtype=torch.float32)
        
        # Use Triton kernel for mean and variance computation
        _batch_norm_mean_var_kernel[1, channels](input_flat, mean, var, batch, height * width, eps)
        
        # Update running statistics
        if running_mean is not None and running_var is not None:
            running_mean.copy_(running_mean * (1 - momentum) + mean * momentum)
            running_var.copy_(running_var * (1 - momentum) + var * momentum)
        
        # Apply batch normalization
        _batch_norm_training_kernel[batch, channels](input_flat, output_flat, mean, var, weight, bias, eps, batch, height * width)
    else:
        # For evaluation mode, use running statistics
        _batch_norm_eval_kernel[batch, channels](input_flat, output_flat, running_mean, running_var, weight, bias, eps, batch, height * width)
    
    return output.view(batch, channels, height, width)

@triton.jit
def _batch_norm_mean_var_kernel(x_ptr, mean_ptr, var_ptr, batch: tl.constexpr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr = 1024):
    # Get channel id
    channel_id = tl.program_id(0)
    
    # Load data for this channel
    offsets = channel_id + tl.arange(0, BLOCK) * tl.num_programs(0)
    mask = offsets < n * batch
    
    # Initialize sum and sum of squares
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Process data in chunks
    for i in range(0, batch * n, BLOCK):
        # Calculate offsets
        chunk_offsets = i + tl.arange(0, BLOCK)
        chunk_mask = chunk_offsets < batch * n
        
        # Load data
        x = tl.load(x_ptr + chunk_offsets, mask=chunk_mask, other=0.0)
        
        # Accumulate sum and sum of squares
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    
    # Compute mean and variance
    total_elements = batch * n
    mean_val = sum_x / total_elements
    var_val = (sum_x2 / total_elements) - (mean_val * mean_val)
    
    # Store results
    tl.store(mean_ptr + channel_id, mean_val)
    tl.store(var_ptr + channel_id, var_val + eps)

@triton.jit
def _batch_norm_training_kernel(x_ptr, y_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, eps: tl.constexpr, batch: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr = 1024):
    # Get batch and channel id
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Load mean and variance for this channel
    mean_val = tl.load(mean_ptr + channel_id)
    var_val = tl.load(var_ptr + channel_id)
    
    # Load weight and bias
    weight_val = tl.load(weight_ptr + channel_id)
    bias_val = tl.load(bias_ptr + channel_id)
    
    # Process data for this batch and channel
    offsets = batch_id + channel_id * batch + tl.arange(0, BLOCK) * batch * channels
    mask = offsets < batch * n
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize and scale
    x_normalized = (x - mean_val) / tl.sqrt(var_val)
    y = weight_val * x_normalized + bias_val
    
    # Store output
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def _batch_norm_eval_kernel(x_ptr, y_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, eps: tl.constexpr, batch: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr = 1024):
    # Get batch and channel id
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Load mean and variance for this channel
    mean_val = tl.load(mean_ptr + channel_id)
    var_val = tl.load(var_ptr + channel_id)
    
    # Load weight and bias
    weight_val = tl.load(weight_ptr + channel_id)
    bias_val = tl.load(bias_ptr + channel_id)
    
    # Process data for this batch and channel
    offsets = batch_id + channel_id * batch + tl.arange(0, BLOCK) * batch * channels
    mask = offsets < batch * n
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize and scale
    x_normalized = (x - mean_val) / tl.sqrt(var_val)
    y = weight_val * x_normalized + bias_val
    
    # Store output
    tl.store(y_ptr + offsets, y, mask=mask)
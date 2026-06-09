import torch
import triton
import triton.language as tl

def _batch_norm_kernel(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr, N: tl.constexpr, C: tl.constexpr, eps: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    # Get the batch and channel dimensions
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Load mean and variance for this channel
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Load weight and bias if provided
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Load input data for this batch and channel
    offsets = batch_idx * C + channel_idx
    x = tl.load(x_ptr + offsets)
    
    # Normalize
    if training:
        # In training mode, use batch statistics
        x_norm = (x - mean) / tl.sqrt(var + eps)
    else:
        # In inference mode, use running statistics
        x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Apply scale and shift
    out = weight * x_norm + bias
    
    # Store result
    tl.store(out_ptr + offsets, out)

@triton.jit
def _batch_norm_train_kernel(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr, N: tl.constexpr, C: tl.constexpr, eps: tl.constexpr, momentum: tl.constexpr, BLOCK: tl.constexpr):
    # Get the batch and channel dimensions
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Load input data for this batch and channel
    offsets = batch_idx * C + channel_idx
    x = tl.load(x_ptr + offsets)
    
    # Load mean and variance for this channel
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Load weight and bias if provided
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Apply scale and shift
    out = weight * x_norm + bias
    
    # Store result
    tl.store(out_ptr + offsets, out)

@triton.jit
def _mean_var_kernel(x_ptr, mean_ptr, var_ptr, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    # Get the channel dimension
    channel_idx = tl.program_id(0)
    
    # Initialize accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Compute sum and sum of squares
    for i in range(0, N, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets * C + channel_idx, mask=mask, other=0.0)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    
    # Compute mean and variance
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    
    # Store results
    tl.store(mean_ptr + channel_idx, mean)
    tl.store(var_ptr + channel_idx, var)

@triton.jit
def _update_running_stats_kernel(mean_ptr, var_ptr, running_mean_ptr, running_var_ptr, momentum: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    # Get the channel dimension
    channel_idx = tl.program_id(0)
    
    # Load current mean and variance
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Load running mean and variance
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    
    # Update running statistics
    new_mean = (1 - momentum) * running_mean + momentum * mean
    new_var = (1 - momentum) * running_var + momentum * var
    
    # Store updated values
    tl.store(running_mean_ptr + channel_idx, new_mean)
    tl.store(running_var_ptr + channel_idx, new_var)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Get input dimensions
    N, C = input.shape[0], input.shape[1]
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Handle training mode
    if training:
        # Compute mean and variance
        mean = torch.empty(C, dtype=torch.float32, device=input.device)
        var = torch.empty(C, dtype=torch.float32, device=input.device)
        
        # Launch kernel to compute mean and variance
        block = 256
        grid = (triton.cdiv(N, block),)
        _mean_var_kernel[grid](input, mean, var, N, C, BLOCK=block)
        
        # Update running statistics if needed
        if weight is not None and bias is not None:
            # Launch kernel to update running statistics
            grid = (C,)
            _update_running_stats_kernel[grid](mean, var, running_mean, running_var, momentum, C, BLOCK=block)
        
        # Launch kernel to compute batch norm
        grid = (N, C)
        _batch_norm_train_kernel[grid](input, mean, var, weight, bias, out, N, C, eps, momentum, BLOCK=block)
    else:
        # Inference mode, use running statistics
        block = 256
        grid = (N, C)
        _batch_norm_kernel[grid](input, running_mean, running_var, weight, bias, out, N, C, eps, training, BLOCK=block)
    
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
#     """
#     Applies Batch Normalization to each channel across a batch of data.
    
#     Parameters:
#         input (Tensor): Input tensor with shape (N, C, H, W) for 4D input (e.g., images).
#         running_mean (Tensor): Running mean for each channel, used in evaluation mode.
#         running_var (Tensor): Running variance for each channel, used in evaluation mode.
#         weight (Tensor, optional): Learnable scaling parameter for each channel.
#         bias (Tensor, optional): Learnable bias for each channel.
#         training (bool): Whether to use the statistics from the current batch or the running statistics.
#         momentum (float): The value used to update running_mean and running_var.
#         eps (float): A small value added to the denominator for numerical stability.

#     Returns:
#         Tensor: The normalized output.
#     """
#     return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def test_batch_norm():
    results = {}

    # Test case 1: Basic test with training=False
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_1"] = batch_norm(input, running_mean, running_var)

    # Test case 2: Test with training=True
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_2"] = batch_norm(input, running_mean, running_var, training=True)

    # Test case 3: Test with weight and bias
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    weight = torch.randn(3, device='cuda')
    bias = torch.randn(3, device='cuda')
    results["test_case_3"] = batch_norm(input, running_mean, running_var, weight, bias)

    # Test case 4: Test with different momentum and eps
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_4"] = batch_norm(input, running_mean, running_var, momentum=0.2, eps=1e-03)

    return results

test_results = test_batch_norm()

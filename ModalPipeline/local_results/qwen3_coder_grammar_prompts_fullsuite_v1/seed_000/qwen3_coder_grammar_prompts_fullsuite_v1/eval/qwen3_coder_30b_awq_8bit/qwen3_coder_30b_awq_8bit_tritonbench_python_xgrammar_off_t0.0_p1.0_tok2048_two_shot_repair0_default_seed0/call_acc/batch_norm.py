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
    
    # Load running mean and variance for this channel
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
    
    # Normalize and apply weight and bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, N * H * W, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N * H * W
        x = tl.load(input_ptr + channel_idx + offsets * C, mask=mask, other=0.0)
        normalized = (x - mean) * inv_std
        if weight_ptr is not None:
            weight = tl.load(weight_ptr + channel_idx)
            normalized = normalized * weight
        if bias_ptr is not None:
            bias = tl.load(bias_ptr + channel_idx)
            normalized = normalized + bias
        tl.store(output_ptr + channel_idx + offsets * C, normalized, mask=mask)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    N, C, H, W = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Handle case where weight and bias are scalars
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor(weight, device=input.device, dtype=input.dtype)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor(bias, device=input.device, dtype=input.dtype)
    
    # Allocate memory for weight and bias if needed
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Launch kernel
    block = 256
    grid = (C,)
    
    # Handle the case where weight and bias are None
    if weight is None:
        weight_ptr = None
    else:
        weight_ptr = weight
    
    if bias is None:
        bias_ptr = None
    else:
        bias_ptr = bias
    
    _batch_norm_kernel[grid](
        input, 
        weight_ptr, 
        bias_ptr, 
        running_mean, 
        running_var,
        output,
        N * H * W,
        C,
        H,
        W,
        training,
        momentum,
        eps,
        BLOCK=block
    )
    
    return output

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

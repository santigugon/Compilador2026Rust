import torch
import triton
import triton.language as tl

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Handle scalar inputs
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if not torch.is_tensor(running_mean):
        running_mean = torch.tensor(running_mean)
    if not torch.is_tensor(running_var):
        running_var = torch.tensor(running_var)
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor(weight)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor(bias)
    
    # Ensure input is 4D (N, C, H, W) for batch norm
    input_shape = input.shape
    if len(input_shape) == 1:
        input = input.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif len(input_shape) == 2:
        input = input.unsqueeze(2).unsqueeze(3)
    elif len(input_shape) == 3:
        input = input.unsqueeze(1)
    
    N, C, H, W = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # For training mode, compute mean and variance
    if training:
        # Compute mean and variance for each channel
        mean = torch.empty(C, dtype=torch.float32, device=input.device)
        var = torch.empty(C, dtype=torch.float32, device=input.device)
        
        # Use PyTorch for mean/var computation in training mode
        for c in range(C):
            channel_data = input[:, c, :, :]
            mean[c] = channel_data.mean()
            var[c] = channel_data.var(unbiased=False)
            
        # Update running statistics
        running_mean.copy_(momentum * mean + (1 - momentum) * running_mean)
        running_var.copy_(momentum * var + (1 - momentum) * running_var)
        
        # Apply batch norm
        input = input.to(torch.float32)
        mean = mean.view(1, C, 1, 1)
        var = var.view(1, C, 1, 1)
        
        # Normalize
        normalized = (input - mean) / torch.sqrt(var + eps)
        
        # Apply weight and bias
        if weight is not None:
            weight = weight.view(1, C, 1, 1)
            normalized = normalized * weight
        
        if bias is not None:
            bias = bias.view(1, C, 1, 1)
            normalized = normalized + bias
        
        output = normalized.to(input.dtype)
    else:
        # Inference mode: use running statistics
        running_mean = running_mean.to(torch.float32)
        running_var = running_var.to(torch.float32)
        
        # Apply batch norm
        input = input.to(torch.float32)
        running_mean = running_mean.view(1, C, 1, 1)
        running_var = running_var.view(1, C, 1, 1)
        
        # Normalize
        normalized = (input - running_mean) / torch.sqrt(running_var + eps)
        
        # Apply weight and bias
        if weight is not None:
            weight = weight.view(1, C, 1, 1)
            normalized = normalized * weight
        
        if bias is not None:
            bias = bias.view(1, C, 1, 1)
            normalized = normalized + bias
        
        output = normalized.to(input.dtype)
    
    # Reshape back to original shape
    if len(input_shape) == 1:
        output = output.squeeze(0).squeeze(2).squeeze(3)
    elif len(input_shape) == 2:
        output = output.squeeze(2).squeeze(3)
    elif len(input_shape) == 3:
        output = output.squeeze(1)
    
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

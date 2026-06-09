import torch
import triton
import triton.language as tl

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
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

    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get output shape
    output_shape = input.shape
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Get input dimensions
    batch_size = input.shape[0]
    channels = input.shape[1]
    
    # Flatten input for easier processing
    input_flat = input.view(batch_size, channels, -1)
    
    # For training mode, we compute batch statistics
    if training:
        # Compute batch mean and variance
        batch_mean = input_flat.mean(dim=(0, 2))
        batch_var = input_flat.var(dim=(0, 2), unbiased=False)
        
        # Update running statistics
        running_mean.copy_(running_mean * (1 - momentum) + batch_mean * momentum)
        running_var.copy_(running_var * (1 - momentum) + batch_var * momentum)
        
        # Normalize
        normalized = (input_flat - batch_mean.view(1, channels, 1)) / (batch_var.view(1, channels, 1) + eps).sqrt()
    else:
        # Use running statistics
        normalized = (input_flat - running_mean.view(1, channels, 1)) / (running_var.view(1, channels, 1) + eps).sqrt()
    
    # Apply scale and shift if provided
    if weight is not None and bias is not None:
        normalized = normalized * weight.view(1, channels, 1) + bias.view(1, channels, 1)
    elif weight is not None:
        normalized = normalized * weight.view(1, channels, 1)
    elif bias is not None:
        normalized = normalized + bias.view(1, channels, 1)
    
    # Reshape back to original shape
    normalized = normalized.view(output_shape)
    
    # Apply SiLU activation
    out = normalized * (1.0 / (1.0 + torch.exp(-normalized)))
    
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_silu_batch_norm():
    results = {}

    # Test case 1: Basic functionality with training=False
    input_tensor = torch.randn(3, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = silu_batch_norm(input_tensor, running_mean, running_var, training=False)

    # Test case 2: With weight and bias, training=False
    weight = torch.ones(5, device='cuda')
    bias = torch.zeros(5, device='cuda')
    results["test_case_2"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=False)

    # Test case 3: With training=True
    results["test_case_3"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True)

    # Test case 4: Different momentum and eps values
    results["test_case_4"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=0.2, eps=1e-3)

    return results

test_results = test_silu_batch_norm()

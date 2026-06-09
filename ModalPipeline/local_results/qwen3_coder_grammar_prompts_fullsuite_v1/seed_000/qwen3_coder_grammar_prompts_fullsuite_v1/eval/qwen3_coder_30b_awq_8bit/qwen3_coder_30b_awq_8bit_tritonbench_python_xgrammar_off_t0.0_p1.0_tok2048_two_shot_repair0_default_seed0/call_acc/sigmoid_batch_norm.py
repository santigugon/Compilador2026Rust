import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_batch_norm_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    n_channels: tl.constexpr,
    eps: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Each block handles one channel
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input for this channel
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Get running statistics for this channel
    mean = tl.load(running_mean_ptr + channel_id, mask=channel_id < n_channels, other=0.0)
    var = tl.load(running_var_ptr + channel_id, mask=channel_id < n_channels, other=0.0)
    
    # Normalize
    normalized = (input - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + channel_id, mask=channel_id < n_channels, other=1.0)
        bias = tl.load(bias_ptr + channel_id, mask=channel_id < n_channels, other=0.0)
        normalized = normalized * weight + bias
    
    # Apply sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-normalized))
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_result, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Handle different input shapes
    if input.dim() == 2:
        N, C = input.shape
        L = 1
    else:  # input.dim() == 3
        N, C, L = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Flatten input for processing
    input_flat = input.view(-1)
    output_flat = output.view(-1)
    
    # Determine block size and grid
    block_size = 256
    n_elements = input_flat.numel()
    
    # Grid configuration: one block per element, one channel per program
    grid = (triton.cdiv(n_elements, block_size), C)
    
    # Launch kernel
    _sigmoid_batch_norm_kernel[grid](
        input_flat,
        running_mean,
        running_var,
        weight,
        bias,
        output_flat,
        n_elements,
        C,
        eps,
        training,
        momentum,
        BLOCK_SIZE=block_size
    )
    
    return output

##################################################################################################################################################



import torch

def test_sigmoid_batch_norm():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(10, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = sigmoid_batch_norm(input_tensor, running_mean, running_var)

    # Test case 2: With learnable parameters (weight and bias)
    weight = torch.ones(5, device='cuda') * 0.5
    bias = torch.zeros(5, device='cuda') + 0.1
    results["test_case_2"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias)

    # Test case 3: In training mode
    results["test_case_3"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, training=True)

    # Test case 4: With a different momentum and eps
    results["test_case_4"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, momentum=0.2, eps=1e-3)

    return results

test_results = test_sigmoid_batch_norm()

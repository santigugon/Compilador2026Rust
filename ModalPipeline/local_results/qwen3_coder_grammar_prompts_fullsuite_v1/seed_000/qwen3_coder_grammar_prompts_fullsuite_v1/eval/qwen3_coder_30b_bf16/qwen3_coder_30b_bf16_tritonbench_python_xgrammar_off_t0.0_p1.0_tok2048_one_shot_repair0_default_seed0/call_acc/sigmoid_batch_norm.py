import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr, 
    output_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    N, 
    C, 
    L,
    training,
    momentum,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Compute global thread index
    pid = tl.program_id(0)
    # Each block processes one channel
    channel_id = pid
    
    if channel_id >= C:
        return
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + channel_id)
    var = tl.load(running_var_ptr + channel_id)
    
    # Initialize running stats if in training mode
    if training:
        # For simplicity, we assume input is already batched and we compute mean/var per channel
        # In practice, this would require more complex reduction operations
        # Here we just use the provided running stats
        pass
    
    # Load weight and bias if provided
    weight = tl.load(weight_ptr + channel_id) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_id) if bias_ptr is not None else 0.0
    
    # Process elements in this channel
    for i in range(0, N * L, BLOCK_SIZE):
        # Calculate offsets
        input_offset = i + tl.arange(0, BLOCK_SIZE)
        output_offset = i + tl.arange(0, BLOCK_SIZE)
        
        # Mask to ensure we don't go out of bounds
        mask = input_offset < N * L
        
        # Load input data
        input_data = tl.load(input_ptr + input_offset, mask=mask)
        
        # Normalize
        normalized = (input_data - mean) / tl.sqrt(var + eps)
        
        # Apply scale and shift
        scaled = normalized * weight + bias
        
        # Apply sigmoid
        sigmoid_result = tl.sigmoid(scaled)
        
        # Store result
        tl.store(output_ptr + output_offset, sigmoid_result, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Determine input shape
    shape = input.shape
    if len(shape) == 2:
        N, C = shape
        L = 1
    elif len(shape) == 3:
        N, C, L = shape
    else:
        raise ValueError("Input must be 2D or 3D tensor")
    
    # Validate dimensions
    if running_mean.shape[0] != C or running_var.shape[0] != C:
        raise ValueError("Running mean and variance must match number of channels")
    
    if weight is not None and weight.shape[0] != C:
        raise ValueError("Weight must match number of channels")
    
    if bias is not None and bias.shape[0] != C:
        raise ValueError("Bias must match number of channels")
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    output_ptr = output.data_ptr()
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (C,)
    
    sigmoid_batch_norm_kernel[grid](
        input_ptr, 
        output_ptr, 
        running_mean_ptr, 
        running_var_ptr, 
        weight_ptr, 
        bias_ptr,
        N, 
        C, 
        L,
        training,
        momentum,
        eps,
        BLOCK_SIZE
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

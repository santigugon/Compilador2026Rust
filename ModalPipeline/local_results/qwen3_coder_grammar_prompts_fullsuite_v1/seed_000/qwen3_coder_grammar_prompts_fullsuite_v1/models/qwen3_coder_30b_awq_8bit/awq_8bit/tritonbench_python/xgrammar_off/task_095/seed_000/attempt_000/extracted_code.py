import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr, 
    output_ptr, 
    mean_ptr, 
    var_ptr, 
    weight_ptr, 
    bias_ptr,
    N, 
    C, 
    H, 
    W,
    training,
    momentum,
    eps,
    BLOCK_SIZE=1024
):
    # Get the channel index for this block
    channel_idx = tl.program_id(0)
    
    # Load mean and variance for this channel
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Load weight and bias if they exist
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Calculate the inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Process elements in chunks
    for i in range(0, N * H * W, BLOCK_SIZE):
        # Calculate the offset for this chunk
        offset = i + tl.arange(0, BLOCK_SIZE)
        
        # Check bounds
        mask = offset < N * H * W
        
        # Load input data
        input_data = tl.load(input_ptr + offset * C + channel_idx, mask=mask)
        
        # Apply batch normalization
        normalized = (input_data - mean) * inv_std
        
        # Apply scale and shift
        output = normalized * weight + bias
        
        # Store result
        tl.store(output_ptr + offset * C + channel_idx, output, mask=mask)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    N, C, H, W = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Handle the case where weight and bias are None
    if weight is None:
        weight = torch.ones(C, device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(C, device=input.device, dtype=input.dtype)
    
    # Ensure weight and bias are contiguous
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Create a grid for the kernel
    grid = (C,)
    
    # Launch kernel
    batch_norm_kernel[grid](
        input,
        output,
        running_mean,
        running_var,
        weight,
        bias,
        N,
        C,
        H,
        W,
        training,
        momentum,
        eps
    )
    
    return output

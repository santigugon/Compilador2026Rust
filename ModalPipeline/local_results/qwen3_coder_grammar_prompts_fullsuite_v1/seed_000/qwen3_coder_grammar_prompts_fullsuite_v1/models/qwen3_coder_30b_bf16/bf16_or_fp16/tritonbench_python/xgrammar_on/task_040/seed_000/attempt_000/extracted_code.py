import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N, C, L,
    training,
    momentum,
    eps,
    BLOCK_SIZE=1024
):
    # Compute global thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Determine the number of elements to process
    num_elements = N * C * L
    
    # Process elements in chunks
    for i in range(block_start, min(block_start + BLOCK_SIZE, num_elements), 1):
        # Calculate indices
        n = i // (C * L)
        c = (i // L) % C
        l = i % L
        
        # Load input
        input_val = tl.load(input_ptr + i)
        
        # Get running mean and variance
        mean_val = tl.load(running_mean_ptr + c)
        var_val = tl.load(running_var_ptr + c)
        
        # Normalize
        normalized = (input_val - mean_val) / tl.sqrt(var_val + eps)
        
        # Apply learnable parameters if provided
        if weight_ptr is not None and bias_ptr is not None:
            weight_val = tl.load(weight_ptr + c)
            bias_val = tl.load(bias_ptr + c)
            normalized = normalized * weight_val + bias_val
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-normalized))
        
        # Store result
        tl.store(output_ptr + i, sigmoid_val)
        
        # Update running statistics if training
        if training:
            # Update mean
            new_mean = (1 - momentum) * mean_val + momentum * input_val
            tl.store(running_mean_ptr + c, new_mean)
            
            # Update variance
            new_var = (1 - momentum) * var_val + momentum * (input_val - mean_val) ** 2
            tl.store(running_var_ptr + c, new_var)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    N, C = input.shape[0], input.shape[1]
    L = input.shape[2] if len(input.shape) > 2 else 1
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    # Launch kernel
    grid = (triton.cdiv(N * C * L, 1024),)
    sigmoid_batch_norm_kernel[grid](
        input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
        output_ptr,
        N, C, L,
        training,
        momentum,
        eps
    )
    
    return output
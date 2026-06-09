import torch
import triton
import triton.language as tl

def _get_block_size(N):
    return min(1024, triton.next_power_of_2(N))

@triton.jit
def _sigmoid_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    training: tl.constexpr, momentum: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the channel index
    c = tl.program_id(0)
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    
    # Compute batch statistics if in training mode
    if training:
        # Compute mean and variance for this channel
        mean = 0.0
        var = 0.0
        
        # Load input data for this channel
        for i in range(0, N * L, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N * L
            
            # Load input values
            input_vals = tl.load(input_ptr + c + offsets * C, mask=mask, other=0.0)
            
            # Accumulate mean
            mean += tl.sum(input_vals)
            
        mean = mean / (N * L)
        
        # Accumulate variance
        for i in range(0, N * L, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N * L
            
            # Load input values
            input_vals = tl.load(input_ptr + c + offsets * C, mask=mask, other=0.0)
            
            # Accumulate variance
            var += tl.sum((input_vals - mean) * (input_vals - mean))
        
        var = var / (N * L)
        
        # Update running statistics
        new_mean = (1 - momentum) * mean + momentum * mean
        new_var = (1 - momentum) * var + momentum * var
        
        tl.store(running_mean_ptr + c, new_mean)
        tl.store(running_var_ptr + c, new_var)
    
    # Normalize and apply sigmoid
    for i in range(0, N * L, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * L
        
        # Load input values
        input_vals = tl.load(input_ptr + c + offsets * C, mask=mask, other=0.0)
        
        # Normalize
        normalized = (input_vals - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias if provided
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + c)
            bias = tl.load(bias_ptr + c)
            normalized = weight * normalized + bias
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-normalized))
        
        # Store output
        tl.store(output_ptr + c + offsets * C, sigmoid_val, mask=mask)

@triton.jit
def _sigmoid_batch_norm_kernel_simple(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    training: tl.constexpr, momentum: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the channel index
    c = tl.program_id(0)
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    
    # Compute batch statistics if in training mode
    if training:
        # Compute mean and variance for this channel
        mean = 0.0
        var = 0.0
        
        # Load input data for this channel
        for i in range(0, N * L, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N * L
            
            # Load input values
            input_vals = tl.load(input_ptr + c + offsets * C, mask=mask, other=0.0)
            
            # Accumulate mean
            mean += tl.sum(input_vals)
            
        mean = mean / (N * L)
        
        # Accumulate variance
        for i in range(0, N * L, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N * L
            
            # Load input values
            input_vals = tl.load(input_ptr + c + offsets * C, mask=mask, other=0.0)
            
            # Accumulate variance
            var += tl.sum((input_vals - mean) * (input_vals - mean))
        
        var = var / (N * L)
        
        # Update running statistics
        new_mean = (1 - momentum) * mean + momentum * mean
        new_var = (1 - momentum) * var + momentum * var
        
        tl.store(running_mean_ptr + c, new_mean)
        tl.store(running_var_ptr + c, new_var)
    
    # Normalize and apply sigmoid
    for i in range(0, N * L, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * L
        
        # Load input values
        input_vals = tl.load(input_ptr + c + offsets * C, mask=mask, other=0.0)
        
        # Normalize
        normalized = (input_vals - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias if provided
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + c)
            bias = tl.load(bias_ptr + c)
            normalized = weight * normalized + bias
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-normalized))
        
        # Store output
        tl.store(output_ptr + c + offsets * C, sigmoid_val, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Handle input shape
    if input.dim() == 2:
        N, C = input.shape
        L = 1
    elif input.dim() == 3:
        N, C, L = input.shape
    else:
        raise ValueError("Input must be 2D or 3D")
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Get block size
    block_size = _get_block_size(N * L)
    
    # Launch kernel
    grid = (C,)
    
    # Handle optional parameters
    weight_ptr = weight if weight is not None else None
    bias_ptr = bias if bias is not None else None
    
    # Call kernel
    _sigmoid_batch_norm_kernel_simple[grid](
        input, running_mean, running_var, weight_ptr, bias_ptr,
        output,
        N, C, L,
        training, momentum, eps,
        block_size
    )
    
    return output
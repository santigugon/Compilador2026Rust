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
    running_mean_ptr,
    running_var_ptr,
    N, 
    C, 
    H, 
    W,
    training,
    momentum,
    eps,
    BLOCK_SIZE
):
    # Get the block index
    block_idx = tl.program_id(0)
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    # Calculate the end position for this block
    end_pos = min(start_pos + BLOCK_SIZE, N * H * W)
    
    # Load input data for this block
    input_block = tl.load(input_ptr + start_pos, mask=start_pos < N * H * W)
    
    # For each channel, compute mean and variance
    for c in range(C):
        # Compute mean for this channel
        mean = tl.sum(input_block) / (N * H * W)
        # Compute variance for this channel
        var = tl.sum((input_block - mean) ** 2) / (N * H * W)
        
        # Update running statistics if training
        if training:
            running_mean = tl.load(running_mean_ptr + c)
            running_var = tl.load(running_var_ptr + c)
            new_mean = (1 - momentum) * running_mean + momentum * mean
            new_var = (1 - momentum) * running_var + momentum * var
            tl.store(running_mean_ptr + c, new_mean)
            tl.store(running_var_ptr + c, new_var)
        else:
            mean = tl.load(mean_ptr + c)
            var = tl.load(var_ptr + c)
        
        # Normalize the input
        normalized = (input_block - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + c)
            bias = tl.load(bias_ptr + c)
            output = normalized * weight + bias
        else:
            output = normalized
            
        # Store the output
        tl.store(output_ptr + start_pos, output, mask=start_pos < N * H * W)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Get the dimensions
    N, C, H, W = input.shape
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Set up the kernel launch parameters
    BLOCK_SIZE = 1024
    num_blocks = (N * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    batch_norm_kernel[
        num_blocks
    ](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        mean_ptr=running_mean.data_ptr() if not training else None,
        var_ptr=running_var.data_ptr() if not training else None,
        weight_ptr=weight.data_ptr() if weight is not None else None,
        bias_ptr=bias.data_ptr() if bias is not None else None,
        running_mean_ptr=running_mean.data_ptr(),
        running_var_ptr=running_var.data_ptr(),
        N=N,
        C=C,
        H=H,
        W=W,
        training=training,
        momentum=momentum,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

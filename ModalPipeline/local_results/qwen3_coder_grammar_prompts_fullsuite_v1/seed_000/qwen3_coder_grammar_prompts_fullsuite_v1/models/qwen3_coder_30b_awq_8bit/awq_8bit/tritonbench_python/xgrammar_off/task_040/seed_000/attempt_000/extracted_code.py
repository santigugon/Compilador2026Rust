import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_batch_norm_kernel(
    input_ptr, 
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles one channel
    channel_id = pid
    
    if channel_id >= C:
        return
        
    # Load running statistics for this channel
    mean_val = tl.load(running_mean_ptr + channel_id)
    var_val = tl.load(running_var_ptr + channel_id)
    
    # Load weight and bias if they exist
    weight_val = tl.load(weight_ptr + channel_id) if weight_ptr is not None else 1.0
    bias_val = tl.load(bias_ptr + channel_id) if bias_ptr is not None else 0.0
    
    # Process elements in this channel
    for i in range(0, N * L, BLOCK):
        # Calculate offsets for this block
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N * L
        
        # Load input values
        input_offsets = channel_id + offsets * C
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x - mean_val) / tl.sqrt(var_val + eps)
        
        # Apply scale and shift
        x_scaled = x_norm * weight_val + bias_val
        
        # Apply sigmoid
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x_scaled))
        
        # Store result
        output_offsets = channel_id + offsets * C
        tl.store(output_ptr + output_offsets, sigmoid_x, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Handle different input shapes
    if input.dim() == 2:
        N, C = input.shape
        L = 1
    elif input.dim() == 3:
        N, C, L = input.shape
    else:
        raise ValueError("Input must be 2D or 3D tensor")
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Determine block size
    BLOCK = 256
    
    # Launch kernel
    grid = (C,)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    output_ptr = out.data_ptr()
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    _sigmoid_batch_norm_kernel[grid](
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
        BLOCK=BLOCK
    )
    
    return out

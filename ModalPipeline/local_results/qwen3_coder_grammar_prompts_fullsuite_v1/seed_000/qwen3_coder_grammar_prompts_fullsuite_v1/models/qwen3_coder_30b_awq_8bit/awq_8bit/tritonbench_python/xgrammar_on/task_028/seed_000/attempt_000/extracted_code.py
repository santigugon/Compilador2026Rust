import torch
import triton
import triton.language as tl

def _silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _silu_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    training: tl.constexpr, momentum: tl.constexpr, eps: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr, input_stride_2: tl.constexpr, input_stride_3: tl.constexpr,
    output_stride_0: tl.constexpr, output_stride_1: tl.constexpr, output_stride_2: tl.constexpr, output_stride_3: tl.constexpr,
    weight_stride_0: tl.constexpr, bias_stride_0: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program handles one channel
    channel_id = pid
    
    if channel_id >= C:
        return
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + channel_id)
    var = tl.load(running_var_ptr + channel_id)
    
    # Compute batch statistics if in training mode
    if training:
        # Compute mean and variance for this channel
        mean = 0.0
        var = 0.0
        
        # Compute mean
        for i in range(0, N * H * W, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N * H * W
            
            # Load input values
            input_offsets = channel_id + offsets * input_stride_1
            x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
            mean += tl.sum(x)
        
        mean = mean / (N * H * W)
        
        # Compute variance
        for i in range(0, N * H * W, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N * H * W
            
            # Load input values
            input_offsets = channel_id + offsets * input_stride_1
            x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
            var += tl.sum((x - mean) * (x - mean))
        
        var = var / (N * H * W)
        
        # Update running statistics
        new_mean = (1 - momentum) * mean + momentum * mean
        new_var = (1 - momentum) * var + momentum * var
        
        tl.store(running_mean_ptr + channel_id, new_mean)
        tl.store(running_var_ptr + channel_id, new_var)
        
        # Update mean and var for normalization
        mean = new_mean
        var = new_var
    
    # Normalize and apply activation
    for i in range(0, N * H * W, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N * H * W
        
        # Load input values
        input_offsets = channel_id + offsets * input_stride_1
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias if provided
        if weight_ptr is not None:
            weight_val = tl.load(weight_ptr + channel_id)
            x_norm = x_norm * weight_val
        
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + channel_id)
            x_norm = x_norm + bias_val
        
        # Apply SiLU activation
        x_silu = _silu(x_norm)
        
        # Store output
        output_offsets = channel_id + offsets * output_stride_1
        tl.store(output_ptr + output_offsets, x_silu, mask=mask)


def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get input dimensions
    N, C, H, W = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Handle optional weight and bias
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Prepare strides
    input_stride_0, input_stride_1, input_stride_2, input_stride_3 = input.stride()
    output_stride_0, output_stride_1, output_stride_2, output_stride_3 = output.stride()
    
    weight_stride_0 = 0
    bias_stride_0 = 0
    if weight is not None:
        weight_stride_0 = weight.stride(0)
    if bias is not None:
        bias_stride_0 = bias.stride(0)
    
    # Launch kernel
    block = 256
    grid = (C,)
    
    _silu_batch_norm_kernel[grid](
        input, running_mean, running_var, weight, bias,
        output,
        N, C, H, W,
        training, momentum, eps,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        weight_stride_0, bias_stride_0,
        BLOCK=block
    )
    
    return output
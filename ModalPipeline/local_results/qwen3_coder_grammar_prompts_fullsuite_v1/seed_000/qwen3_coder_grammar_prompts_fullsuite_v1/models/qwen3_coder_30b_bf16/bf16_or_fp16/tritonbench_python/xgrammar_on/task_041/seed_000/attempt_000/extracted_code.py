import torch
import triton
import triton.language as tl

@triton.jit
def _fused_hardsigmoid_batch_norm_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor
    running_mean_ptr,  # pointer to running mean
    running_var_ptr,  # pointer to running variance
    N,  # number of elements
    C,  # number of channels
    training,  # training mode flag
    momentum,  # momentum for running stats
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Get the channel index
    channel_idx = tl.program_id(0)
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    
    # Load weight and bias if they exist
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Initialize accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Load data in chunks
    for i in range(0, N, BLOCK_SIZE):
        # Calculate offsets
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        
        # Load input data
        x = tl.load(x_ptr + channel_idx + offset * C, mask=mask)
        
        # Accumulate sum and sum of squares
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
        
    # Compute mean and variance
    if training:
        mean = sum_x / N
        var = (sum_x2 / N) - (mean * mean)
        
        # Update running statistics
        new_mean = (1 - momentum) * mean + momentum * tl.load(running_mean_ptr + channel_idx)
        new_var = (1 - momentum) * var + momentum * tl.load(running_var_ptr + channel_idx)
        
        tl.store(running_mean_ptr + channel_idx, new_mean)
        tl.store(running_var_ptr + channel_idx, new_var)
    
    # Normalize and apply activation
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        
        # Load input
        x = tl.load(x_ptr + channel_idx + offset * C, mask=mask)
        
        # Normalize
        x_norm = (x - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias
        x_norm = x_norm * weight + bias
        
        # Apply Hardsigmoid
        hardsigmoid = tl.where(x_norm >= 3.0, 1.0, tl.where(x_norm <= -3.0, 0.0, (x_norm + 3.0) / 6.0))
        
        # Store output
        tl.store(out_ptr + channel_idx + offset * C, hardsigmoid, mask=mask)


def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False
) -> torch.Tensor:
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    N, C = x.shape[0], x.shape[1]
    
    # Create output tensor
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
    
    # Launch kernel
    grid = (C,)
    BLOCK_SIZE = 1024
    
    _fused_hardsigmoid_batch_norm_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        weight_ptr=weight.data_ptr() if weight is not None else None,
        bias_ptr=bias.data_ptr() if bias is not None else None,
        running_mean_ptr=running_mean.data_ptr(),
        running_var_ptr=running_var.data_ptr(),
        N=N,
        C=C,
        training=training,
        momentum=momentum,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
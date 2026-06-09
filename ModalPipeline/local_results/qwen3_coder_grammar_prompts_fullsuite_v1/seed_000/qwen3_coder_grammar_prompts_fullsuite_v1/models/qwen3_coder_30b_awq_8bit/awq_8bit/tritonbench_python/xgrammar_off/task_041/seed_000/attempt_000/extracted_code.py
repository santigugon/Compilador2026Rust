import torch
import triton
import triton.language as tl

@triton.jit
def _batch_norm_hardsigmoid_kernel(
    x_ptr, 
    out_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    n: tl.constexpr,
    c: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    cid = tl.program_id(1)
    
    # Each block handles one channel
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input data
    x = tl.load(x_ptr + cid * n + offsets, mask=mask, other=0.0)
    
    # Batch normalization
    if training:
        # Compute mean and variance for this batch
        mean = tl.sum(x) / n
        var = tl.sum((x - mean) * (x - mean)) / n
        # Update running statistics
        running_mean = tl.load(running_mean_ptr + cid)
        running_var = tl.load(running_var_ptr + cid)
        new_mean = (1 - momentum) * running_mean + momentum * mean
        new_var = (1 - momentum) * running_var + momentum * var
        tl.store(running_mean_ptr + cid, new_mean)
        tl.store(running_var_ptr + cid, new_var)
        # Normalize
        x_norm = (x - mean) / tl.sqrt(var + eps)
    else:
        # Use running statistics
        mean = tl.load(running_mean_ptr + cid)
        var = tl.load(running_var_ptr + cid)
        x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + cid)
        bias = tl.load(bias_ptr + cid)
        x_norm = x_norm * weight + bias
    
    # Apply Hardsigmoid: min(1, max(0, (x + 3) / 6))
    hardsigmoid = tl.minimum(1.0, tl.maximum(0.0, (x_norm + 3.0) / 6.0))
    
    # Store result
    tl.store(out_ptr + cid * n + offsets, hardsigmoid, mask=mask)

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
    # Handle the case where x is a scalar
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0]
    channels = x.shape[1]
    n = batch_size * channels
    
    # Create output tensor
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
    
    # Handle case where we have a single channel
    if channels == 1:
        # Use a simpler kernel for single channel
        block = 256
        grid = (triton.cdiv(n, block), 1)
        
        @triton.jit
        def _single_channel_kernel(
            x_ptr, 
            out_ptr,
            weight_ptr,
            bias_ptr,
            running_mean_ptr,
            running_var_ptr,
            n: tl.constexpr,
            training: tl.constexpr,
            momentum: tl.constexpr,
            eps: tl.constexpr,
            BLOCK: tl.constexpr
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            
            # Load input data
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # Batch normalization
            if training:
                # Compute mean and variance for this batch
                mean = tl.sum(x) / n
                var = tl.sum((x - mean) * (x - mean)) / n
                # Update running statistics
                running_mean = tl.load(running_mean_ptr)
                running_var = tl.load(running_var_ptr)
                new_mean = (1 - momentum) * running_mean + momentum * mean
                new_var = (1 - momentum) * running_var + momentum * var
                tl.store(running_mean_ptr, new_mean)
                tl.store(running_var_ptr, new_var)
                # Normalize
                x_norm = (x - mean) / tl.sqrt(var + eps)
            else:
                # Use running statistics
                mean = tl.load(running_mean_ptr)
                var = tl.load(running_var_ptr)
                x_norm = (x - mean) / tl.sqrt(var + eps)
            
            # Apply weight and bias if provided
            if weight_ptr is not None and bias_ptr is not None:
                weight = tl.load(weight_ptr)
                bias = tl.load(bias_ptr)
                x_norm = x_norm * weight + bias
            
            # Apply Hardsigmoid: min(1, max(0, (x + 3) / 6))
            hardsigmoid = tl.minimum(1.0, tl.maximum(0.0, (x_norm + 3.0) / 6.0))
            
            # Store result
            tl.store(out_ptr + offsets, hardsigmoid, mask=mask)
        
        _single_channel_kernel[grid](
            x, 
            out,
            weight,
            bias,
            running_mean,
            running_var,
            n,
            training,
            momentum,
            eps,
            BLOCK=block
        )
    else:
        # Use multi-channel kernel
        block = 256
        grid = (triton.cdiv(n, block), channels)
        
        # Handle None weights and biases
        weight_ptr = weight if weight is not None else None
        bias_ptr = bias if bias is not None else None
        
        _batch_norm_hardsigmoid_kernel[grid](
            x, 
            out,
            weight_ptr,
            bias_ptr,
            running_mean,
            running_var,
            n,
            channels,
            training,
            momentum,
            eps,
            BLOCK=block
        )
    
    return out

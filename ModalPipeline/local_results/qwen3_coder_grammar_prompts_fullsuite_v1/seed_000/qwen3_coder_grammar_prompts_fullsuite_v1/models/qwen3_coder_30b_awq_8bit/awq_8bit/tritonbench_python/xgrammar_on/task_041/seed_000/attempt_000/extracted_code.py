import torch
import triton
import triton.language as tl

def _hardsigmoid(x):
    return tl.maximum(0.0, tl.minimum(1.0, (x + 1.0) / 2.0))

@triton.jit
def _fused_batch_norm_hardsigmoid_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the column index
    col = tl.program_id(0)
    
    if col >= C:
        return
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + col)
    var = tl.load(running_var_ptr + col)
    
    # Initialize accumulators for mean and variance computation
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Compute mean and variance if in training mode
    if training:
        # Load data for this channel
        for i in range(0, N, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x_vals = tl.load(x_ptr + col + offsets * C, mask=mask, other=0.0)
            sum_x += tl.sum(x_vals)
            sum_x2 += tl.sum(x_vals * x_vals)
        
        # Compute mean and variance
        mean = sum_x / N
        var = sum_x2 / N - mean * mean
        
        # Update running statistics
        new_mean = (1.0 - momentum) * mean + momentum * mean
        new_var = (1.0 - momentum) * var + momentum * var
        
        tl.store(running_mean_ptr + col, new_mean)
        tl.store(running_var_ptr + col, new_var)
    
    # Normalize and apply activation
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(x_ptr + col + offsets * C, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x_vals - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias if provided
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + col)
            bias = tl.load(bias_ptr + col)
            x_norm = x_norm * weight + bias
        
        # Apply hardsigmoid
        hardsigmoid_val = _hardsigmoid(x_norm)
        
        # Store result
        tl.store(output_ptr + col + offsets * C, hardsigmoid_val, mask=mask)

@triton.jit
def _fused_batch_norm_hardsigmoid_kernel_simple(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the column index
    col = tl.program_id(0)
    
    if col >= C:
        return
    
    # Load running mean and variance for this channel
    mean = tl.load(running_mean_ptr + col)
    var = tl.load(running_var_ptr + col)
    
    # Compute mean and variance if in training mode
    if training:
        # Load data for this channel
        x_vals = tl.load(x_ptr + col, mask=tl.arange(0, N) < N, other=0.0)
        # Compute mean and variance
        mean = tl.sum(x_vals) / N
        var = tl.sum(x_vals * x_vals) / N - mean * mean
        
        # Update running statistics
        new_mean = (1.0 - momentum) * mean + momentum * mean
        new_var = (1.0 - momentum) * var + momentum * var
        
        tl.store(running_mean_ptr + col, new_mean)
        tl.store(running_var_ptr + col, new_var)
    
    # Normalize and apply activation
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(x_ptr + col + offsets * C, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x_vals - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias if provided
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + col)
            bias = tl.load(bias_ptr + col)
            x_norm = x_norm * weight + bias
        
        # Apply hardsigmoid
        hardsigmoid_val = _hardsigmoid(x_norm)
        
        # Store result
        tl.store(output_ptr + col + offsets * C, hardsigmoid_val, mask=mask)

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
        output = x
    else:
        output = torch.empty_like(x)
    
    # Handle case where weight and bias are None
    if weight is None:
        weight = torch.ones(C, device=x.device, dtype=x.dtype)
    if bias is None:
        bias = torch.zeros(C, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    block_size = 256
    grid_size = triton.cdiv(C, block_size)
    
    # Use a simple kernel for now
    _fused_batch_norm_hardsigmoid_kernel_simple[
        (grid_size,)
    ](
        x, running_mean, running_var, weight, bias,
        output,
        N, C,
        training,
        momentum,
        eps,
        block_size
    )
    
    return output
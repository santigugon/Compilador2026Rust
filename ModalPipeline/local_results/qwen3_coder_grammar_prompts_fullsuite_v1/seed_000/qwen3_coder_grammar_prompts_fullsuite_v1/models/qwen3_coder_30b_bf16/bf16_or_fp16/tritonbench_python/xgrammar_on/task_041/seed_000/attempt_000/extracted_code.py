import torch
import triton
import triton.language as tl

def _hardsigmoid(x):
    return tl.maximum(0.0, tl.minimum(1.0, (x + 1.0) / 6.0))

@triton.jit
def _fused_hardsigmoid_batch_norm_kernel(
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
    
    # Load weight and bias if they exist
    weight = tl.load(weight_ptr + col) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + col) if bias_ptr is not None else 0.0
    
    # Compute batch mean and variance
    if training:
        # Compute mean
        mean_sum = 0.0
        for i in range(0, N, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x_vals = tl.load(x_ptr + col + offsets * C, mask=mask, other=0.0)
            mean_sum += tl.sum(x_vals)
        mean = mean_sum / N
        
        # Compute variance
        var_sum = 0.0
        for i in range(0, N, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x_vals = tl.load(x_ptr + col + offsets * C, mask=mask, other=0.0)
            diff = x_vals - mean
            var_sum += tl.sum(diff * diff)
        var = var_sum / N
        
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
        normalized = (x_vals - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias
        if weight_ptr is not None:
            normalized = normalized * weight
        if bias_ptr is not None:
            normalized = normalized + bias
        
        # Apply Hardsigmoid
        hardsigmoid_out = _hardsigmoid(normalized)
        
        # Store result
        tl.store(output_ptr + col + offsets * C, hardsigmoid_out, mask=mask)


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
        weight_ptr = None
    else:
        weight_ptr = weight
        
    if bias is None:
        bias_ptr = None
    else:
        bias_ptr = bias
    
    # Launch kernel
    block_size = 256
    grid_size = triton.cdiv(C, block_size)
    
    _fused_hardsigmoid_batch_norm_kernel[grid_size](
        x, running_mean, running_var, weight_ptr, bias_ptr,
        output,
        N, C,
        training,
        momentum,
        eps,
        BLOCK_SIZE=block_size
    )
    
    return output
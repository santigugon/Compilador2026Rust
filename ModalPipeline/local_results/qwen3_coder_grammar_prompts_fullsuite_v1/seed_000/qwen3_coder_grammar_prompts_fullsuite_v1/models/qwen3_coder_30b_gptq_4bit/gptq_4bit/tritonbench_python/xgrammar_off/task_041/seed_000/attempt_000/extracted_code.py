import torch
import triton
import triton.language as tl

@triton.jit
def _batch_norm_hardsigmoid_kernel(
    x_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    output_ptr,
    n_features: tl.constexpr,
    n_samples: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    training: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_features
    
    # Load running mean and variance
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
    
    # Compute batch mean and variance if in training mode
    if training:
        # Compute mean and variance for each feature
        batch_mean = tl.sum(tl.load(x_ptr + offsets, mask=mask, other=0.0)) / n_samples
        batch_var = tl.sum((tl.load(x_ptr + offsets, mask=mask, other=0.0) - batch_mean) ** 2) / n_samples
        
        # Update running statistics
        updated_mean = (1 - momentum) * running_mean + momentum * batch_mean
        updated_var = (1 - momentum) * running_var + momentum * batch_var
        
        # Store updated running statistics
        tl.store(running_mean_ptr + offsets, updated_mean, mask=mask)
        tl.store(running_var_ptr + offsets, updated_var, mask=mask)
        
        # Use batch statistics
        mean = batch_mean
        var = batch_var
    else:
        # Use running statistics
        mean = running_mean
        var = running_var
    
    # Normalize
    x_normalized = (tl.load(x_ptr + offsets, mask=mask, other=0.0) - mean) / (tl.sqrt(var + eps))
    
    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        x_normalized = x_normalized * weight + bias
    
    # Apply Hardsigmoid
    # Hardsigmoid: min(max((x + 3) / 6, 0), 1)
    hardsigmoid = tl.minimum(tl.maximum((x_normalized + 3.0) / 6.0, 0.0), 1.0)
    
    # Store result
    tl.store(output_ptr + offsets, hardsigmoid, mask=mask)

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
    n_samples = x.shape[0]
    n_features = x.shape[1]
    
    # Create output tensor
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
    
    # Handle scalar weight and bias
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor(weight, device=x.device, dtype=x.dtype)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor(bias, device=x.device, dtype=x.dtype)
    
    # Prepare pointers
    x_ptr = x.data_ptr()
    output_ptr = out.data_ptr()
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    BLOCK = 256
    grid = (triton.cdiv(n_features, BLOCK),)
    
    _batch_norm_hardsigmoid_kernel[grid](
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        n_features,
        n_samples,
        momentum,
        eps,
        training,
        BLOCK=BLOCK
    )
    
    return out

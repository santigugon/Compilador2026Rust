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
    eps: tl.constexpr,
    momentum: tl.constexpr,
    training: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    feature_id = pid
    
    if feature_id >= n_features:
        return
    
    # Load running statistics
    mean = tl.load(running_mean_ptr + feature_id)
    var = tl.load(running_var_ptr + feature_id)
    
    # Load input for this feature across all samples
    x_offsets = feature_id + tl.arange(0, BLOCK) * n_features
    x = tl.load(x_ptr + x_offsets, mask=tl.arange(0, BLOCK) < n_samples, other=0.0)
    
    # Batch normalization
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + feature_id)
        bias = tl.load(bias_ptr + feature_id)
        x_norm = x_norm * weight + bias
    
    # Hardsigmoid: min(max(0, x + 3) / 6, 1)
    x_hardsigmoid = tl.minimum(tl.maximum(x_norm + 3.0, 0.0) / 6.0, 1.0)
    
    # Store output
    output_offsets = feature_id + tl.arange(0, BLOCK) * n_features
    tl.store(output_ptr + output_offsets, x_hardsigmoid, mask=tl.arange(0, BLOCK) < n_samples)

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
    # Handle inplace operation
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
    
    # Ensure input is contiguous for easier memory access
    x = x.contiguous()
    out = out.contiguous()
    
    # Get dimensions
    n_samples, n_features = x.shape[0], x.shape[1]
    
    # Create output tensor
    if not inplace:
        out = torch.empty_like(x)
    else:
        out = x
    
    # Set up kernel launch parameters
    block = 256
    grid = (triton.cdiv(n_features, 1),)
    
    # Handle case where weight and bias are None
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    _batch_norm_hardsigmoid_kernel[grid](
        x,
        running_mean,
        running_var,
        weight_ptr,
        bias_ptr,
        out,
        n_features,
        n_samples,
        eps,
        momentum,
        training,
        BLOCK=block
    )
    
    return out

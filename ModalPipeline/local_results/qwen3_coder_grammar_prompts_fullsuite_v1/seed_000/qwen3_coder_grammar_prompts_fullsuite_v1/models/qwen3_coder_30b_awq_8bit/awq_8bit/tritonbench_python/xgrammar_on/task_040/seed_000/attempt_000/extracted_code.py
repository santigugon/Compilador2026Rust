import torch
import triton
import triton.language as tl

def _get_block_size(n):
    return min(1024, triton.next_power_of_2(n))

@triton.jit
def _sigmoid_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    n_features: tl.constexpr,
    n_samples: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the feature index
    feature_idx = tl.program_id(0)
    
    if feature_idx >= n_features:
        return
    
    # Load running mean and variance for this feature
    mean = tl.load(running_mean_ptr + feature_idx)
    var = tl.load(running_var_ptr + feature_idx)
    
    # Initialize output tensor
    output = tl.zeros((n_samples,), dtype=tl.float32)
    
    # Process samples in blocks
    for i in range(0, n_samples, BLOCK_SIZE):
        # Calculate offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_samples
        
        # Load input values
        input_vals = tl.load(input_ptr + feature_idx + offsets * n_features, mask=mask, other=0.0)
        
        # Normalize
        normalized = (input_vals - mean) / tl.sqrt(var + eps)
        
        # Apply weight and bias if provided
        if weight_ptr is not None:
            weight_val = tl.load(weight_ptr + feature_idx)
            normalized = normalized * weight_val
        
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + feature_idx)
            normalized = normalized + bias_val
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-normalized))
        
        # Store result
        tl.store(output_ptr + feature_idx + offsets * n_features, sigmoid_val, mask=mask)
    
    # Update running statistics if training
    if training:
        # Compute batch mean and variance
        batch_mean = tl.sum(input_vals) / n_samples
        batch_var = tl.sum((input_vals - batch_mean) * (input_vals - batch_mean)) / n_samples
        
        # Update running mean and variance
        new_mean = (1 - momentum) * mean + momentum * batch_mean
        new_var = (1 - momentum) * var + momentum * batch_var
        
        tl.store(running_mean_ptr + feature_idx, new_mean)
        tl.store(running_var_ptr + feature_idx, new_var)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    if input.dim() == 2:
        n_samples, n_features = input.shape
    elif input.dim() == 3:
        n_samples, n_features, _ = input.shape
    else:
        raise ValueError("Input must be 2D or 3D")
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Handle case where weight and bias are scalars
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor(weight, device=input.device, dtype=input.dtype)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor(bias, device=input.device, dtype=input.dtype)
    
    # Get pointers
    input_ptr = input.data_ptr()
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    output_ptr = output.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    block_size = _get_block_size(n_samples)
    grid_size = (n_features,)
    
    _sigmoid_batch_norm_kernel[grid_size](
        input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
        output_ptr,
        n_features, n_samples, training, momentum, eps, block_size
    )
    
    return output
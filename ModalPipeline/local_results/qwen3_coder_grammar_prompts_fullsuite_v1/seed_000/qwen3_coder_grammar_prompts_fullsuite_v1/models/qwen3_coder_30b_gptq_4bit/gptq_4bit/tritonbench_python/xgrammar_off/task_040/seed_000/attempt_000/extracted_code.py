import torch
import triton
import triton.language as tl

@triton.jit
def _batch_norm_sigmoid_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    n_features: tl.constexpr,
    n_samples: tl.constexpr,
    eps: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_features
    
    if training:
        # Compute mean and variance for this batch
        input_block = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        mean = tl.sum(input_block) / n_samples
        var = tl.sum((input_block - mean) * (input_block - mean)) / n_samples
        
        # Update running statistics
        running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
        running_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
        updated_mean = (1 - momentum) * running_mean + momentum * mean
        updated_var = (1 - momentum) * running_var + momentum * var
        tl.store(running_mean_ptr + offsets, updated_mean, mask=mask)
        tl.store(running_var_ptr + offsets, updated_var, mask=mask)
        
        # Normalize
        normalized = (input_block - mean) / tl.sqrt(var + eps)
    else:
        # Use running statistics
        running_mean_block = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
        running_var_block = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
        normalized = (input_block - running_mean_block) / tl.sqrt(running_var_block + eps)
    
    # Apply weight and bias
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        normalized = normalized * weight + bias
    
    # Apply sigmoid
    output = 1.0 / (1.0 + tl.exp(-normalized))
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Handle different input shapes
    if input.dim() == 2:
        N, C = input.shape
        L = 1
    else:
        N, C, L = input.shape
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Handle weight and bias
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get the number of features (channels)
    n_features = C
    n_samples = N * L
    
    # Determine block size
    BLOCK = 256
    
    # Launch kernel
    grid = (triton.cdiv(n_features, BLOCK),)
    
    # Create a temporary tensor for the input to pass to the kernel
    input_ptr = input.data_ptr()
    output_ptr = out.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    
    # Launch kernel
    _batch_norm_sigmoid_kernel[grid](
        input_ptr,
        output_ptr,
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        n_features,
        n_samples,
        eps,
        training,
        momentum,
        BLOCK=BLOCK
    )
    
    return out

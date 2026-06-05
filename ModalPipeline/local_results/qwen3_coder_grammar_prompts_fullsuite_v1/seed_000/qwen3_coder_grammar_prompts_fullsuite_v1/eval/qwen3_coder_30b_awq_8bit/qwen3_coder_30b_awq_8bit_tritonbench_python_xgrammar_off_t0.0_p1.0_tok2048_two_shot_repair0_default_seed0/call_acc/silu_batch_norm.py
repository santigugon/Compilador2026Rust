import torch
import triton
import triton.language as tl

@triton.jit
def _silu_batch_norm_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    output_ptr,
    n_features: tl.constexpr,
    n_samples: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    # Get the feature dimension (channels)
    pid = tl.program_id(0)
    feature_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = feature_offsets < n_features
    
    # Load input for this feature across all samples
    input_feature = tl.load(input_ptr + feature_offsets, mask=mask, other=0.0)
    
    # Load running statistics
    mean = tl.load(running_mean_ptr + feature_offsets, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + feature_offsets, mask=mask, other=0.0)
    
    # Normalize
    normalized = (input_feature - mean) / tl.sqrt(var + eps)
    
    # Apply scale and shift if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + feature_offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + feature_offsets, mask=mask, other=0.0)
        normalized = normalized * weight + bias
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-normalized))
    silu_output = normalized * sigmoid_x
    
    # Store result
    tl.store(output_ptr + feature_offsets, silu_output, mask=mask)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous for easier handling
    input = input.contiguous()
    
    # Get dimensions
    n_samples, n_features = input.shape[0], input.shape[1]
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Determine block size
    BLOCK = 256
    
    # Launch kernel
    grid = (triton.cdiv(n_features, BLOCK),)
    
    # Handle optional weight and bias
    weight_ptr = weight if weight is not None else None
    bias_ptr = bias if bias is not None else None
    
    # Call kernel
    _silu_batch_norm_kernel[grid](
        input,
        running_mean,
        running_var,
        weight_ptr,
        bias_ptr,
        output,
        n_features,
        n_samples,
        training,
        momentum,
        eps,
        BLOCK=BLOCK
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_silu_batch_norm():
    results = {}

    # Test case 1: Basic functionality with training=False
    input_tensor = torch.randn(3, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = silu_batch_norm(input_tensor, running_mean, running_var, training=False)

    # Test case 2: With weight and bias, training=False
    weight = torch.ones(5, device='cuda')
    bias = torch.zeros(5, device='cuda')
    results["test_case_2"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=False)

    # Test case 3: With training=True
    results["test_case_3"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True)

    # Test case 4: Different momentum and eps values
    results["test_case_4"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=0.2, eps=1e-3)

    return results

test_results = test_silu_batch_norm()

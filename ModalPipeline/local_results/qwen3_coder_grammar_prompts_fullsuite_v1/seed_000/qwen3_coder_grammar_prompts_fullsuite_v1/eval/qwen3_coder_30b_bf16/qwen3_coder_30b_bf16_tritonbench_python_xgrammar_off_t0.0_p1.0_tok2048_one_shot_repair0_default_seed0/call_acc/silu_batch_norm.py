import torch
import triton
import triton.language as tl

@triton.jit
def silu_batch_norm_kernel(
    input_ptr, 
    output_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    N,
    training,
    momentum,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_block = tl.load(input_ptr + offset, mask=offset < N)
    
    # Batch normalization
    if training:
        mean = tl.sum(input_block) / N
        var = tl.sum((input_block - mean) ** 2) / N
        running_mean = tl.load(running_mean_ptr)
        running_var = tl.load(running_var_ptr)
        new_mean = (1 - momentum) * running_mean + momentum * mean
        new_var = (1 - momentum) * running_var + momentum * var
        tl.store(running_mean_ptr, new_mean)
        tl.store(running_var_ptr, new_var)
    else:
        mean = tl.load(running_mean_ptr)
        var = tl.load(running_var_ptr)
    
    # Normalize
    normalized = (input_block - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr)
        bias = tl.load(bias_ptr)
        normalized = normalized * weight + bias
    
    # Apply SiLU activation
    output_block = normalized * tl.sigmoid(normalized)
    
    tl.store(output_ptr + offset, output_block, mask=offset < N)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    assert input.dim() == 2, "Input must be 2D tensor"
    assert input.shape[1] == running_mean.shape[0] == running_var.shape[0], "Channel dimension mismatch"
    
    if weight is not None:
        assert weight.shape[0] == running_mean.shape[0], "Weight dimension mismatch"
    if bias is not None:
        assert bias.shape[0] == running_mean.shape[0], "Bias dimension mismatch"
    
    output = torch.empty_like(input)
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    output_ptr = output.data_ptr()
    running_mean_ptr = running_mean.data_ptr()
    running_var_ptr = running_var.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    silu_batch_norm_kernel[grid](
        input_ptr,
        output_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        N,
        training,
        momentum,
        eps,
        BLOCK_SIZE
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

import torch
import triton
import triton.language as tl

@triton.jit
def silu_batch_norm_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    output_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    input = tl.load(input_ptr + offsets, mask=mask)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask)
    running_var = tl.load(running_var_ptr + offsets, mask=mask)
    
    # Batch normalization
    normalized = (input - running_mean) / tl.sqrt(running_var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask)
        bias = tl.load(bias_ptr + offsets, mask=mask)
        normalized = normalized * weight + bias
    
    # Apply SiLU activation
    output = normalized * tl.sigmoid(normalized)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    assert input.shape == running_mean.shape == running_var.shape, "Input and running statistics must have the same shape"
    
    if weight is not None:
        assert weight.shape == input.shape, "Weight must have the same shape as input"
    if bias is not None:
        assert bias.shape == input.shape, "Bias must have the same shape as input"
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Prepare output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    silu_batch_norm_kernel[grid](
        input_ptr=input.data_ptr(),
        running_mean_ptr=running_mean.data_ptr(),
        running_var_ptr=running_var.data_ptr(),
        weight_ptr=weight.data_ptr() if weight is not None else None,
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=output.data_ptr(),
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

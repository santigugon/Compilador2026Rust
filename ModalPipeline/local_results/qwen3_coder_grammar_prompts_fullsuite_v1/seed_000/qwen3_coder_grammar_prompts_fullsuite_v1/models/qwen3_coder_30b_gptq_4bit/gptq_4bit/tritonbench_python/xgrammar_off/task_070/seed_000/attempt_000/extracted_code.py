import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    X_ptr, Y_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, G, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    
    # Element-wise addition and multiplication
    xy = x + y
    xy = xy * y
    
    # Group normalization
    # Reshape for group normalization
    xy_reshaped = xy.reshape(N, G, C // G)
    
    # Compute mean and variance for each group
    group_means = tl.sum(xy_reshaped, axis=2) / (C // G)
    group_means = tl.broadcast_to(group_means, xy_reshaped.shape)
    
    group_vars = tl.sum((xy_reshaped - group_means) ** 2, axis=2) / (C // G)
    group_vars = tl.broadcast_to(group_vars, xy_reshaped.shape)
    
    # Normalize
    xy_normalized = (xy_reshaped - group_means) / tl.sqrt(group_vars + eps)
    
    # Scale and shift
    weight = tl.load(weight_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    bias = tl.load(bias_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    
    weight = tl.broadcast_to(weight, xy_reshaped.shape)
    bias = tl.broadcast_to(bias, xy_reshaped.shape)
    
    output = xy_normalized * weight + bias
    
    # Reshape back to original shape
    output = output.reshape(N, C)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    # Validate inputs
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert weight.shape == (input1.shape[-1],), "weight must have shape (C,)"
    assert bias.shape == (input1.shape[-1],), "bias must have shape (C,)"
    assert input1.shape[-1] % num_groups == 0, "Number of channels must be divisible by num_groups"
    
    # Flatten input tensors
    N = input1.numel()
    C = input1.shape[-1]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input1)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    fused_add_mul_groupnorm_kernel[grid](
        input1, input2, weight, bias, out,
        N, C, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

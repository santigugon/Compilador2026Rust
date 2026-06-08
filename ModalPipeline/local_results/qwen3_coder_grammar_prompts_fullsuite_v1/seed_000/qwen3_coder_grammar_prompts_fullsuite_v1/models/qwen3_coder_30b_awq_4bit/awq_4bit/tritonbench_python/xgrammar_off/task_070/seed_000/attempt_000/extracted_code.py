import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    X_ptr, Y_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, G, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # Load data
    X = tl.load(X_ptr + start_pos, mask=(start_pos + tl.arange(0, BLOCK_SIZE) < N))
    Y = tl.load(Y_ptr + start_pos, mask=(start_pos + tl.arange(0, BLOCK_SIZE) < N))
    output = tl.load(output_ptr + start_pos, mask=(start_pos + tl.arange(0, BLOCK_SIZE) < N))
    
    # Element-wise addition and multiplication
    result = (X + Y) * output
    
    # Group normalization
    # For simplicity, we assume C is divisible by G
    group_size = C // G
    group_idx = (start_pos // C) % G
    
    # Calculate mean and variance for the group
    mean = tl.sum(result, axis=0) / group_size
    var = tl.sum((result - mean) ** 2, axis=0) / group_size
    
    # Normalize
    normalized = (result - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + (start_pos % C), mask=(start_pos % C < C))
    bias = tl.load(bias_ptr + (start_pos % C), mask=(start_pos % C < C))
    
    output = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + start_pos, output, mask=(start_pos + tl.arange(0, BLOCK_SIZE) < N))

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Check shapes
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert weight.shape == (input1.shape[1],), "weight must have shape (C,)"
    assert bias.shape == (input1.shape[1],), "bias must have shape (C,)"
    
    # Flatten inputs for easier processing
    input1_flat = input1.view(-1)
    input2_flat = input2.view(-1)
    output_flat = input1_flat + input2_flat
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input1)
    else:
        assert out.shape == input1.shape, "out must have the same shape as input1"
    
    # Flatten output for processing
    out_flat = out.view(-1)
    
    # Get total number of elements
    N = input1_flat.numel()
    C = input1.shape[1]
    G = num_groups
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_mul_groupnorm_kernel[
        num_blocks,
        1,
        (N, C, G, eps),
        BLOCK_SIZE
    ](
        input1_flat,
        input2_flat,
        weight,
        bias,
        out_flat,
        N, C, G, eps,
        BLOCK_SIZE
    )
    
    return out.view(input1.shape)

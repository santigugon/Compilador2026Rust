import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, G,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get the thread index
    pid = tl.program_id(0)
    # Each block processes one group
    group_id = pid
    
    # Calculate the start and end indices for this group
    group_size = C // G
    start = group_id * group_size
    end = start + group_size
    
    # Shared memory for mean and std calculation
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load data for this group
    x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE) * C + start)
    y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE) * C + start)
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE) + start)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE) + start)
    
    # Element-wise addition and multiplication
    z = (x + y) * y
    
    # Compute mean and variance for this group
    mean = tl.sum(z, axis=0) / N
    var = tl.sum((z - mean) ** 2, axis=0) / N
    
    # Normalize
    normalized = (z - mean) / tl.sqrt(var + eps)
    
    # Apply scale and shift
    output = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + tl.arange(0, BLOCK_SIZE) * C + start, output)

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    # Validate inputs
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert weight.shape == bias.shape, "weight and bias must have the same shape"
    assert len(weight.shape) == 1, "weight must be 1D"
    assert weight.shape[0] == input1.shape[1], "weight must have the same number of channels as input1"
    
    # Get dimensions
    N, C = input1.shape
    G = num_groups
    
    # Ensure C is divisible by G
    assert C % G == 0, "Number of channels must be divisible by number of groups"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input1)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (G,)
    
    fused_add_mul_groupnorm_kernel[grid](
        input1, input2, weight, bias, out,
        N, C, G,
        eps,
        BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_add_mul_groupnorm():
    results = {}

    # Test case 1: Basic functionality test
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 2
    results["test_case_1"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 2: Different shapes for input1 and input2 (broadcastable)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(1, 4, 1, 1, device='cuda')  # Broadcastable shape
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 2
    results["test_case_2"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 3: Single group normalization (equivalent to layer normalization)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 1
    results["test_case_3"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 4: No weight and bias (should default to None)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    num_groups = 2
    results["test_case_4"] = fused_add_mul_groupnorm(input1, input2, None, None, num_groups)

    return results

test_results = test_fused_add_mul_groupnorm()

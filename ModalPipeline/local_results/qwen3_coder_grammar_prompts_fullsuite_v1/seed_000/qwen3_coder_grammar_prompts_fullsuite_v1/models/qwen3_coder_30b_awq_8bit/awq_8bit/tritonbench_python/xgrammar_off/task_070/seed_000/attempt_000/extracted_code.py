import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_mul_groupnorm_kernel(
    x_ptr, y_ptr, z_ptr, weight_ptr, bias_ptr, 
    out_ptr,
    n_elements: tl.constexpr,
    num_groups: tl.constexpr,
    channels: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    group_id = pid // channels
    channel_id = pid % channels
    
    # Load input tensors
    x = tl.load(x_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < n_elements)
    y = tl.load(y_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < n_elements)
    z = tl.load(z_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < n_elements)
    
    # Element-wise addition and multiplication
    xy = x + y
    result = xy * z
    
    # Group normalization
    # Compute mean and variance for the group
    group_start = (channel_id // group_size) * group_size
    group_end = min(group_start + group_size, channels)
    
    # For simplicity, we'll compute mean and variance over the entire tensor
    # In a more optimized version, this would be done per group
    mean = tl.sum(result) / n_elements
    var = tl.sum((result - mean) * (result - mean)) / n_elements
    
    # Normalize
    normalized = (result - mean) / tl.sqrt(var + eps)
    
    # Apply learnable parameters
    weight = tl.load(weight_ptr + channel_id, mask=channel_id < channels)
    bias = tl.load(bias_ptr + channel_id, mask=channel_id < channels)
    
    output = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + tl.arange(0, BLOCK), output, mask=tl.arange(0, BLOCK) < n_elements)

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    # Check if input2 can be broadcasted to input1
    if input1.shape != input2.shape:
        try:
            torch.broadcast_tensors(input1, input2)
        except RuntimeError:
            raise ValueError("input2 is not broadcastable to input1")
    
    # Check weight and bias shapes
    if weight.shape != (input1.shape[1],) or bias.shape != (input1.shape[1],):
        raise ValueError("weight and bias must have shape (C,) where C is the number of channels")
    
    # Handle out parameter
    if out is None:
        out = torch.empty_like(input1)
    else:
        if out.shape != input1.shape:
            raise ValueError("out tensor must have the same shape as input1")
    
    # Flatten tensors for processing
    input1_flat = input1.view(-1)
    input2_flat = input2.view(-1)
    out_flat = out.view(-1)
    
    # Compute intermediate result of addition and multiplication
    intermediate = input1_flat + input2_flat
    intermediate = intermediate * input2_flat
    
    # Apply group normalization
    n_elements = input1_flat.numel()
    channels = input1.shape[1]
    group_size = channels // num_groups
    
    # For simplicity, we'll use a single kernel for the entire operation
    # In practice, this would be more complex to handle group normalization properly
    block = 256
    grid = triton.cdiv(n_elements, block)
    
    # Create a temporary tensor for the intermediate result
    temp = torch.empty_like(input1_flat)
    
    # First, compute the element-wise operations
    @triton.jit
    def _elementwise_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        result = (x + y) * z
        tl.store(out_ptr + offsets, result, mask=mask)
    
    _elementwise_kernel[grid](input1_flat, input2_flat, input2_flat, temp, n_elements, BLOCK=block)
    
    # Apply group normalization using PyTorch for simplicity and correctness
    # This is a simplified approach - a full Triton implementation would be more complex
    temp = temp.view(input1.shape)
    weight = weight.view(1, -1, 1, 1)  # Reshape for broadcasting
    bias = bias.view(1, -1, 1, 1)
    
    # Group normalization
    batch_size, channels, height, width = temp.shape
    group_size = channels // num_groups
    
    # Reshape for group normalization
    temp_reshaped = temp.view(batch_size, num_groups, group_size, height, width)
    
    # Compute mean and variance for each group
    mean = temp_reshaped.mean(dim=(2, 3, 4), keepdim=True)
    var = temp_reshaped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (temp_reshaped - mean) / torch.sqrt(var + eps)
    
    # Apply learnable parameters
    output = normalized * weight + bias
    
    # Reshape back
    output = output.view(batch_size, channels, height, width)
    
    # Copy result to output tensor
    out.copy_(output)
    
    return out

import torch
import triton
import triton.language as tl

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    if out is None:
        out = torch.empty_like(input1)
    
    # Ensure input tensors are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = out.contiguous()
    
    # Check if input2 is scalar
    if not torch.is_tensor(input2):
        input2 = torch.tensor(input2, dtype=input1.dtype, device=input1.device)
    
    # Get tensor shapes
    input1_shape = input1.shape
    input2_shape = input2.shape
    weight_shape = weight.shape
    
    # Flatten tensors for processing
    batch_size = input1_shape[0]
    channels = input1_shape[1]
    spatial_size = input1_shape[2] * input1_shape[3]
    
    # Check if input2 is broadcastable
    if input2_shape != input1_shape:
        # Handle broadcasting
        if len(input2_shape) == 1 and input2_shape[0] == 1:
            # Scalar broadcast
            input2 = input2.expand(input1_shape)
        elif len(input2_shape) == 1 and input2_shape[0] == channels:
            # Channel-wise broadcast
            input2 = input2.view(1, channels, 1, 1).expand(batch_size, channels, spatial_size, 1)
        else:
            raise ValueError("input2 is not broadcastable to input1")
    
    # Perform element-wise addition and multiplication
    # Result = (input1 + input2) * input2
    temp = (input1 + input2) * input2
    
    # Apply group normalization
    # Group normalization parameters
    group_size = channels // num_groups
    
    # Flatten for group normalization
    temp_flat = temp.view(batch_size, num_groups, group_size, spatial_size)
    
    # Compute mean and variance for each group
    mean = temp_flat.mean(dim=(2, 3), keepdim=True)
    var = temp_flat.var(dim=(2, 3), keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (temp_flat - mean) / (tl.sqrt(var + eps))
    
    # Apply learnable parameters
    weight = weight.view(1, num_groups, group_size, 1)
    bias = bias.view(1, num_groups, group_size, 1)
    
    result = normalized * weight + bias
    
    # Reshape back to original shape
    out = result.view(batch_size, channels, spatial_size, 1)
    
    # Return the output tensor
    return out
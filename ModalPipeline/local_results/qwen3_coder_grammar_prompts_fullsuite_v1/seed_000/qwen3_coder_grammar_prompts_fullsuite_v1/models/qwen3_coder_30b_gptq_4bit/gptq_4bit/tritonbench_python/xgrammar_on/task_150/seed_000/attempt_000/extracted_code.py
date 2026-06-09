import torch
import triton
import triton.language as tl

def broadcast_tensors(*tensors):
    if len(tensors) == 0:
        return []
    
    # Find the target shape by computing the broadcasted shape
    target_shape = torch.broadcast_shapes(*[t.shape for t in tensors])
    
    # Broadcast each tensor to the target shape
    result = []
    for tensor in tensors:
        if tensor.shape == target_shape:
            result.append(tensor)
        else:
            # Create output tensor with target shape
            out = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
            # Use PyTorch's built-in broadcasting for the actual operation
            result.append(tensor.expand(target_shape))
    
    return result
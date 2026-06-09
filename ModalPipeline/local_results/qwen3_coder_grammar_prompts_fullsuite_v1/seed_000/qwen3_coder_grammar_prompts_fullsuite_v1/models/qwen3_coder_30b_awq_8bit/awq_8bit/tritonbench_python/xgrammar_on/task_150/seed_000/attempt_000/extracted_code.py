import torch
import triton
import triton.language as tl

def broadcast_tensors(*tensors):
    if len(tensors) == 0:
        return []
    
    # Get the target shape by broadcasting all input shapes
    target_shape = torch.broadcast_shapes(*[t.shape for t in tensors])
    
    # Create output tensors with the target shape
    result = []
    for tensor in tensors:
        if tensor.shape == target_shape:
            # No broadcasting needed
            result.append(tensor)
        else:
            # Create a new tensor with the target shape
            out = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
            # Use PyTorch's broadcast_to for the actual broadcasting
            # This is a safe approach since we're using PyTorch's implementation
            # for the complex broadcasting logic
            result.append(torch.broadcast_to(tensor, target_shape))
    
    return result
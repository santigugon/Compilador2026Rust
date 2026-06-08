import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    input_ptr, 
    output_ptr, 
    size, 
    stride, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    input_ptrs = input_ptr + offsets * stride
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, tl.load(input_ptrs, mask=mask), mask=mask)

def broadcast_tensors(*tensors):
    # Convert to torch tensors if needed
    torch_tensors = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in tensors]
    
    # Get the maximum shape through broadcasting
    shapes = [t.shape for t in torch_tensors]
    max_shape = torch.broadcast_shapes(*shapes)
    
    # Create output tensors with the broadcasted shape
    result = []
    for tensor in torch_tensors:
        # Create a new tensor with the broadcasted shape
        broadcasted = tensor.expand(max_shape)
        result.append(broadcasted)
    
    return result

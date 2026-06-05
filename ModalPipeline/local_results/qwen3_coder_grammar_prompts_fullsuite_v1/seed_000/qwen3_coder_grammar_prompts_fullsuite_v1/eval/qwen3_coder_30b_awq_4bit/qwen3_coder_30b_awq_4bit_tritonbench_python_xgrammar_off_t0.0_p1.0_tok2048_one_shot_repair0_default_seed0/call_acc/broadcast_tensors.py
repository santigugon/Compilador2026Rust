import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    input_ptr, 
    output_ptr, 
    size, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)

def broadcast_tensors(*tensors):
    # Convert to torch tensors if needed
    torch_tensors = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in tensors]
    
    # Get the maximum shape through broadcasting
    shapes = [t.shape for t in torch_tensors]
    max_shape = torch.broadcast_shapes(*shapes)
    
    # Create output tensors with the broadcasted shape
    output_tensors = []
    
    for tensor in torch_tensors:
        # Create a new tensor with the broadcasted shape
        if tensor.shape == max_shape:
            output_tensors.append(tensor)
        else:
            # Create a new tensor with the broadcasted shape
            expanded_tensor = tensor.expand(max_shape)
            output_tensors.append(expanded_tensor)
    
    return output_tensors

##################################################################################################################################################



import torch

def test_broadcast_tensors():
    results = {}

    # Test case 1: Broadcasting a scalar and a 1D tensor
    x1 = torch.tensor(3.0, device='cuda')
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = broadcast_tensors(x1, y1)

    # Test case 2: Broadcasting two 1D tensors of different sizes
    x2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y2 = torch.tensor([1.0], device='cuda')
    results["test_case_2"] = broadcast_tensors(x2, y2)

    # Test case 3: Broadcasting a 2D tensor and a 1D tensor
    x3 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_3"] = broadcast_tensors(x3, y3)

    # Test case 4: Broadcasting two 2D tensors of different shapes
    x4 = torch.tensor([[1.0], [2.0], [3.0]], device='cuda')
    y4 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    results["test_case_4"] = broadcast_tensors(x4, y4)

    return results

test_results = test_broadcast_tensors()

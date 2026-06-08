import torch
import triton
import triton.language as tl

@triton.jit
def _index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape0: tl.constexpr, input_shape1: tl.constexpr,
    index_size: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Compute the total number of elements in the output
    total_elements = input_shape0 * index_size
    
    # Mask for valid elements
    mask = offsets < total_elements
    
    # Calculate which index we're working with
    index_id = offsets // input_shape0
    input_id = offsets % input_shape0
    
    # Load index values
    index_val = tl.load(index_ptr + index_id, mask=index_id < index_size, other=0)
    
    # Load input values
    input_val = tl.load(input_ptr + input_id * input_shape1 + index_val, mask=mask, other=0.0)
    
    # Load other values (scalar or tensor)
    other_val = tl.load(other_ptr + input_id, mask=mask, other=0.0)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if dim < 0:
        dim = input.dim() + dim
    
    if dim >= input.dim():
        raise ValueError("dim must be within the range of input tensor dimensions")
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other is the same dtype as input
    if other.dtype != input.dtype:
        other = other.to(input.dtype)
    
    # Ensure other is on the same device as input
    if other.device != input.device:
        other = other.to(input.device)
    
    # Get output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.size(0)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError("out tensor must have the same shape as the selected elements")
        if out.dtype != torch.bool:
            raise ValueError("out tensor must have bool dtype")
    
    # Handle the case where we need to broadcast other
    if other.numel() == 1:
        # Scalar case - broadcast to match output shape
        other = other.expand(output_shape)
    else:
        # Ensure other has the right shape for broadcasting
        other = other.expand(output_shape)
    
    # Flatten input and other for easier indexing
    input_flat = input.view(input.shape[0], -1)
    other_flat = other.view(output_shape[0], -1)
    
    # Get the size of the dimension we're indexing along
    index_size = index.size(0)
    
    # Calculate total elements in output
    total_elements = output_shape[0] * index_size
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(total_elements, block),)
    
    # Flatten the tensors for kernel execution
    input_flat = input.view(-1)
    other_flat = other.view(-1)
    index_flat = index.view(-1)
    
    _index_select_eq_kernel[grid](
        input_flat, index_flat, other_flat, out.view(-1),
        output_shape[0], input.shape[1], index_size, BLOCK=block
    )
    
    return out

##################################################################################################################################################



import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()

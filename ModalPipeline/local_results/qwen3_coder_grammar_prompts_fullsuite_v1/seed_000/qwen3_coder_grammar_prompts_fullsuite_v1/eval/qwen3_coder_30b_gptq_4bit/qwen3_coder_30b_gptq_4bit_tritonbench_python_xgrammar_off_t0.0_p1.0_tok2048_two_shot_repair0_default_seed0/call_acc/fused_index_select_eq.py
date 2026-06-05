import torch
import triton
import triton.language as tl

@triton.jit
def _index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape0: tl.constexpr, input_shape1: tl.constexpr,
    index_size: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Calculate total elements in output
    total_elements = input_shape0 * input_shape1
    
    # Create mask for valid indices
    mask = offsets < total_elements
    
    # For simplicity, we'll handle the indexing in the main function
    # and just do the comparison in the kernel
    if dim == 0:
        # Select elements along dim 0
        input_offsets = offsets
        other_offsets = offsets
    else:
        # Select elements along dim 1
        input_offsets = offsets
        other_offsets = offsets
    
    # Load input and other values
    input_val = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    other_val = tl.load(other_ptr + other_offsets, mask=mask, other=0.0)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if dim < 0:
        dim = input.dim() + dim
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other is the same device and dtype as input
    if other.device != input.device:
        other = other.to(input.device)
    if other.dtype != input.dtype:
        other = other.to(input.dtype)
    
    # Get output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.size(0)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError("Output tensor shape does not match expected shape")
        if out.dtype != torch.bool:
            raise ValueError("Output tensor must be boolean type")
    
    # Handle the case where we need to do index selection
    # For simplicity, we'll use PyTorch's native implementation for index selection
    # and only use Triton for the comparison part
    
    # Select elements using PyTorch's index_select
    selected = torch.index_select(input, dim, index)
    
    # Perform element-wise equality comparison using Triton
    n = selected.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create temporary tensors for Triton kernel
    selected_flat = selected.view(-1)
    other_flat = other.view(-1)
    out_flat = out.view(-1)
    
    # Ensure other is broadcastable
    if other_flat.size(0) == 1:
        # Scalar case - broadcast to match selected
        other_flat = other_flat.expand(selected_flat.size(0))
    elif other_flat.size(0) != selected_flat.size(0):
        # If other is not scalar and not matching, we need to handle differently
        # For now, we'll assume they are compatible or other is scalar
        pass
    
    # Launch kernel
    _index_select_eq_kernel[grid](
        selected_flat, index, other_flat, out_flat,
        selected_flat.size(0), 1,  # For simplicity, assuming 1D indexing
        index.size(0), dim, BLOCK=block
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

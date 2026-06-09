import torch
import triton
import triton.language as tl

def _get_total_elements(tensors):
    total = 0
    for t in tensors:
        total += t.numel()
    return total

def _get_output_shape(tensors):
    # Get the shape of the first tensor for reference
    first_shape = tensors[0].shape
    # All tensors must have the same shape except for the last dimension
    # which can be different for hstacking
    if len(tensors) == 1:
        return first_shape
    
    # Check if all tensors have compatible shapes for hstack
    # All dimensions except the last must be the same
    for t in tensors:
        if t.shape[:-1] != first_shape[:-1]:
            raise ValueError("All tensors must have compatible shapes for hstack")
    
    # For hstack, we concatenate along the last dimension
    total_last_dim = sum(t.shape[-1] for t in tensors)
    return first_shape[:-1] + (total_last_dim,)

@triton.jit
def _hstack_div_kernel(
    stacked_ptr, divisor_ptr, out_ptr,
    total_elements: tl.constexpr,
    rounding_mode: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    
    stacked = tl.load(stacked_ptr + offsets, mask=mask, other=0.0)
    divisor = tl.load(divisor_ptr + offsets, mask=mask, other=1.0)
    
    # Perform division
    result = stacked / divisor
    
    # Apply rounding if specified
    if rounding_mode == "trunc":
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == "floor":
        result = tl.floor(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _hstack_kernel(
    input_ptrs, output_ptr,
    total_elements: tl.constexpr,
    tensor_shapes: tl.constexpr,
    tensor_strides: tl.constexpr,
    tensor_offsets: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    
    # This is a simplified version - in practice, we would need to
    # properly handle the offsets for each tensor
    # For now, we'll assume a simple case
    tl.store(output_ptr + offsets, tl.load(input_ptrs[0] + offsets, mask=mask, other=0.0), mask=mask)


def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Validate inputs
    if not tensors:
        raise ValueError("tensors must not be empty")
    
    # Get the output shape
    output_shape = _get_output_shape(tensors)
    
    # Create the stacked tensor
    stacked = torch.cat(tensors, dim=-1)
    
    # Handle divisor
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=stacked.dtype, device=stacked.device)
    
    # Ensure divisor is broadcastable
    if divisor.shape != stacked.shape:
        try:
            divisor = divisor.expand(stacked.shape)
        except RuntimeError:
            raise ValueError("divisor is not broadcastable to the stacked tensor shape")
    
    # Perform the division
    if out is not None:
        result = out
    else:
        result = torch.empty_like(stacked)
    
    # Apply rounding if specified
    if rounding_mode is None:
        result = stacked / divisor
    elif rounding_mode == "trunc":
        result = torch.trunc(stacked / divisor)
    elif rounding_mode == "floor":
        result = torch.floor(stacked / divisor)
    else:
        raise ValueError("Invalid rounding_mode. Must be None, 'trunc', or 'floor'")
    
    # If out is provided, we need to copy the result to out
    if out is not None:
        out.copy_(result)
        return out
    
    return result
##################################################################################################################################################



import torch

def test_fused_hstack_div():
    results = {}

    # Test case 1: Basic functionality with two tensors and a scalar divisor
    tensors1 = [torch.tensor([1, 2], device='cuda'), torch.tensor([3, 4], device='cuda')]
    divisor1 = 2
    results["test_case_1"] = fused_hstack_div(tensors1, divisor1)

    # Test case 3: Using rounding_mode='floor'
    tensors3 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor3 = 2
    results["test_case_3"] = fused_hstack_div(tensors3, divisor3, rounding_mode='floor')

    # Test case 4: Using rounding_mode='trunc'
    tensors4 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor4 = 2
    results["test_case_4"] = fused_hstack_div(tensors4, divisor4, rounding_mode='trunc')

    return results

test_results = test_fused_hstack_div()

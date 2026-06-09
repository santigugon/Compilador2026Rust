import torch
import triton
import triton.language as tl

def _fused_hstack_div_kernel(
    input_ptr, divisor_ptr, output_ptr,
    input_size, divisor_size, output_size,
    rounding_mode,
    BLOCK_SIZE: int = 1024
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, output_size)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Load input and divisor
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    divisor_vals = tl.load(divisor_ptr + offsets, mask=mask)
    
    # Perform division
    result = input_vals / divisor_vals
    
    # Apply rounding if specified
    if rounding_mode == 'trunc':
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 'floor':
        result = tl.floor(result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def _fused_hstack_div_kernel_triton(
    input_ptr, divisor_ptr, output_ptr,
    input_size, divisor_size, output_size,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, output_size)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Load input and divisor
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    divisor_vals = tl.load(divisor_ptr + offsets, mask=mask)
    
    # Perform division
    result = input_vals / divisor_vals
    
    # Apply rounding if specified
    if rounding_mode == 'trunc':
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 'floor':
        result = tl.floor(result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Flatten all tensors and stack them horizontally
    stacked = torch.cat(tensors, dim=-1)
    
    # Handle divisor
    if isinstance(divisor, (int, float)):
        divisor_tensor = torch.tensor(divisor, dtype=stacked.dtype, device=stacked.device)
    else:
        divisor_tensor = divisor
    
    # Broadcast divisor to match stacked tensor
    if divisor_tensor.shape != stacked.shape:
        divisor_tensor = divisor_tensor.expand_as(stacked)
    
    # Perform division
    if rounding_mode is None:
        result = stacked / divisor_tensor
    elif rounding_mode == 'trunc':
        result = torch.trunc(stacked / divisor_tensor)
    elif rounding_mode == 'floor':
        result = torch.floor(stacked / divisor_tensor)
    else:
        raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")
    
    # Handle output tensor
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

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_hstack_div_kernel(
    stacked_ptr, 
    divisor_ptr, 
    out_ptr, 
    n: tl.constexpr, 
    rounding_mode: tl.constexpr, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    stacked = tl.load(stacked_ptr + offsets, mask=mask, other=0.0)
    divisor = tl.load(divisor_ptr + offsets, mask=mask, other=1.0)
    
    # Perform division
    result = stacked / divisor
    
    # Apply rounding if specified
    if rounding_mode == 'trunc':
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 'floor':
        result = tl.floor(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Handle scalar divisor
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=torch.float32, device=torch.device('cuda'))
    
    # Stack tensors horizontally
    stacked = torch.hstack(tensors)
    
    # Handle output tensor
    if out is not None:
        out = out
    else:
        out = torch.empty_like(stacked)
    
    # Determine the number of elements
    n = stacked.numel()
    
    # Set block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle rounding mode
    rounding_mode_enum = None
    if rounding_mode == 'trunc':
        rounding_mode_enum = 'trunc'
    elif rounding_mode == 'floor':
        rounding_mode_enum = 'floor'
    
    # Launch kernel
    _fused_hstack_div_kernel[grid](
        stacked, 
        divisor, 
        out, 
        n, 
        rounding_mode_enum, 
        BLOCK=block
    )
    
    return out

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

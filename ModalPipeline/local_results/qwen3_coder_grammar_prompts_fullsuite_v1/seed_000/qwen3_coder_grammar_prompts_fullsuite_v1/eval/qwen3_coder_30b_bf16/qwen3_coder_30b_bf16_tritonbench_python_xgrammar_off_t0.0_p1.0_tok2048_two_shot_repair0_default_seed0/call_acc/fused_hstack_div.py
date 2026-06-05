import torch
import triton
import triton.language as tl

@triton.jit
def _fused_hstack_div_kernel(
    stacked_ptr, 
    divisor_ptr, 
    out_ptr, 
    n_elements: tl.constexpr, 
    rounding_mode: tl.constexpr,
    stacked_stride: tl.constexpr,
    divisor_stride: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    
    stacked = tl.load(stacked_ptr + offsets * stacked_stride, mask=mask, other=0.0)
    divisor = tl.load(divisor_ptr + offsets * divisor_stride, mask=mask, other=1.0)
    
    # Perform division
    result = stacked / divisor
    
    # Apply rounding if specified
    if rounding_mode == 0:  # None
        pass  # No rounding needed
    elif rounding_mode == 1:  # trunc
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 2:  # floor
        result = tl.floor(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    if not tensors:
        raise ValueError("tensors must not be empty")
    
    # Stack tensors horizontally
    stacked = torch.hstack(tensors)
    
    # Handle divisor (could be tensor or scalar)
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=stacked.dtype, device=stacked.device)
    
    # Ensure divisor is broadcastable to stacked
    if divisor.shape == ():
        divisor = divisor.expand_as(stacked)
    else:
        # Broadcast divisor to match stacked shape
        try:
            torch.broadcast_tensors(stacked, divisor)
        except RuntimeError:
            raise ValueError("divisor is not broadcastable to stacked tensor")
    
    # Determine output tensor
    if out is not None:
        if out.shape != stacked.shape:
            raise ValueError("out tensor must have the same shape as the stacked tensor")
        out = out
    else:
        out = torch.empty_like(stacked)
    
    # Determine rounding mode
    rounding_mode_enum = 0  # None
    if rounding_mode == 'trunc':
        rounding_mode_enum = 1
    elif rounding_mode == 'floor':
        rounding_mode_enum = 2
    
    # Launch kernel
    n = stacked.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _fused_hstack_div_kernel[grid](
        stacked, 
        divisor, 
        out, 
        n, 
        rounding_mode_enum,
        stacked.stride(0) if stacked.dim() > 0 else 1,
        divisor.stride(0) if divisor.dim() > 0 else 1,
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

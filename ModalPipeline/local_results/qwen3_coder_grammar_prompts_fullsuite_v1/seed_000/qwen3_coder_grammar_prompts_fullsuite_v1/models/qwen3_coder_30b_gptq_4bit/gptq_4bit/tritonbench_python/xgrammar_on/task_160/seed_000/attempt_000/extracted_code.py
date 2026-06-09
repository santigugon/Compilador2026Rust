import torch
import triton
import triton.language as tl

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Handle the case where dtype is None (default to input's dtype)
    if dtype is None:
        dtype = input.dtype
    
    # Handle the case where device is None (default to input's device)
    if device is None:
        device = input.device
    
    # Handle the case where layout is None (default to input's layout)
    if layout is None:
        layout = input.layout
    
    # Create output tensor with same shape as input
    out = torch.empty_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    
    # If the tensor is empty, return it immediately
    if out.numel() == 0:
        return out
    
    # For non-empty tensors, fill with ones using Triton kernel
    n = out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Define the kernel to fill with ones
    @triton.jit
    def _fill_ones_kernel(out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(out_ptr + offsets, 1.0, mask=mask)
    
    # Launch the kernel
    _fill_ones_kernel[grid](out, n, BLOCK=block)
    
    return out
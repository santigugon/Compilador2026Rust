import torch
import triton
import triton.language as tl

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if not torch.is_tensor(index):
        raise TypeError("index must be a tensor")
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other_tensor = other
    
    # Perform index selection
    selected = torch.index_select(input, dim, index)
    
    # Perform element-wise equality comparison
    if out is not None:
        result = out
        if result.shape != selected.shape:
            raise ValueError("Output tensor shape must match selected tensor shape")
    else:
        result = torch.empty_like(selected, dtype=torch.bool)
    
    # Use Triton kernel for element-wise equality
    _eq_kernel[triton.cdiv(selected.numel(), 256)](selected, other_tensor, result, selected.numel())
    
    return result

@triton.jit
def _eq_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr = 256):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x == y
    tl.store(out_ptr + offsets, result, mask=mask)
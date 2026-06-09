import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_dot_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_new = y + alpha * x
    tl.store(y_ptr + offsets, y_new, mask=mask)
    # Compute dot product of modified y with itself
    y_new_squared = y_new * y_new
    # Reduce to get sum of squares
    sum_val = tl.sum(y_new_squared, axis=0)
    # Store the result
    tl.store(out_ptr, sum_val, mask=True)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure inputs are contiguous and have the same shape
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=y.device)
    
    # Get total number of elements
    n = y.numel()
    
    # Set block size
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    
    # Launch kernel
    _scaled_add_dot_kernel[grid](y, x, out, n, alpha, BLOCK=BLOCK)
    
    return out

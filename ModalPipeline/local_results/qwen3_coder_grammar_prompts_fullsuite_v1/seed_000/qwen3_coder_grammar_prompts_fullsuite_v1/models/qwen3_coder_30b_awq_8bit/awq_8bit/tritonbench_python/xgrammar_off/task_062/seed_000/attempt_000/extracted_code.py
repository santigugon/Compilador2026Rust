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
    y_modified = y + alpha * x
    tl.store(y_ptr + offsets, y_modified, mask=mask)
    # Compute dot product of modified y with itself
    tl.store(out_ptr, tl.sum(y_modified * y_modified), mask=True)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    y = y.contiguous()
    x = x.contiguous()
    
    # Check shapes
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.numel() > 0, "y and x must not be empty"
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=y.device)
    
    # Launch kernel
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _scaled_add_dot_kernel[grid](y, x, out, n, alpha, BLOCK=block)
    
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_norm_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = y + alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)
    # Compute squared sum for norm
    y_squared = y * y
    tl.atomic_add(out_ptr, y_squared, mask=mask)

def scaled_add_norm(y, x, alpha):
    # Ensure inputs are contiguous
    y = y.contiguous()
    x = x.contiguous()
    
    # Create output tensor for squared sum
    out = torch.zeros(1, dtype=torch.float32, device=y.device)
    
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _scaled_add_norm_kernel[grid](y, x, out, n, alpha, BLOCK=block)
    
    # Compute the 2-norm
    norm = torch.sqrt(out[0])
    return norm

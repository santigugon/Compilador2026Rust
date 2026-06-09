import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot_kernel(y_ptr, x_ptr, out_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < n
    y = tl.load(y_ptr + offset, mask=mask)
    x = tl.load(x_ptr + offset, mask=mask)
    y_new = y + alpha * x
    tl.store(y_ptr + offset, y_new, mask=mask)
    # Compute dot product of modified y with itself
    dot_product = tl.sum(y_new * y_new)
    if pid == 0:
        tl.store(out_ptr, dot_product)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.dtype == x.dtype, "y and x must have the same dtype"
    n = y.numel()
    out = torch.empty(1, dtype=y.dtype, device=y.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scaled_add_dot_kernel[grid](y, x, out, n, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out

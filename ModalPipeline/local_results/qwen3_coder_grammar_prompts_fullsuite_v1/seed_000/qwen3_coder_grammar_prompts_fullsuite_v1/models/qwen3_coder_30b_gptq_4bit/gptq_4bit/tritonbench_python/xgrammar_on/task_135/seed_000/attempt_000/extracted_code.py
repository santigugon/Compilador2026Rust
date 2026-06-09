import torch
import triton
import triton.language as tl

@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x * y
    tl.store(out_ptr + offsets, result, mask=mask)


def mul(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if input.shape != out.shape:
            raise ValueError("Input and output tensors must have the same shape.")

    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)

    if not torch.is_tensor(other):
        # Handle scalar multiplication
        other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
        other = other_tensor

    _mul_kernel[grid](input, other, out, n, BLOCK=block)
    return out
import torch
import triton
import triton.language as tl

@triton.jit
def _pow_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, broadcast_y: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if broadcast_y:
        y = tl.load(y_ptr, mask=True, other=0.0)
        y = tl.broadcast_to(y, (BLOCK,))
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    result = tl.libdevice.pow(x, y)
    tl.store(out_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    input = input.contiguous()
    
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.contiguous()
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if torch.is_tensor(exponent):
        exponent = exponent.contiguous()
        broadcast_y = False
        # Check if shapes are broadcastable
        try:
            torch.broadcast_tensors(input, exponent)
        except RuntimeError:
            raise ValueError("Input and exponent shapes are not broadcastable")
    else:
        broadcast_y = True
        exponent = torch.tensor(exponent, dtype=input.dtype, device=input.device)
    
    _pow_kernel[grid](input, exponent, out, n, broadcast_y, BLOCK=block)
    return out

import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input - alpha * other
    tl.store(output_ptr + offsets, output, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    if input.dtype != other.dtype:
        other = other.to(input.dtype)
    
    if input.shape != other.shape:
        # Broadcasting is handled by PyTorch's native implementation
        # We'll use the standard torch.sub for broadcasting
        return torch.sub(input, other, alpha=alpha, out=out)
    
    if out is None:
        out = torch.empty_like(input)
    
    size = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    _sub_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        alpha=alpha,
        size=size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

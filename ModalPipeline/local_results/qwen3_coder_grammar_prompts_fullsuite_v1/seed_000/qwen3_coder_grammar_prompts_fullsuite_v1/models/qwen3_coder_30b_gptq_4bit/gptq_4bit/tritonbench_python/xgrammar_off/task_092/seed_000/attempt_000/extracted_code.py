import torch
import triton
import triton.language as tl

@triton.jit
def _tensordot_rsqrt_kernel(
    a_ptr, b_ptr, out_ptr,
    a_size, b_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    a_offsets = tl.arange(offset, offset + BLOCK_SIZE)
    b_offsets = tl.arange(offset, offset + BLOCK_SIZE)
    
    a_mask = a_offsets < a_size
    b_mask = b_offsets < b_size
    
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask)
    b_vals = tl.load(b_ptr + b_offsets, mask=b_mask)
    
    # Element-wise reciprocal square root
    out_vals = 1.0 / tl.sqrt(a_vals * b_vals)
    tl.store(out_ptr + a_offsets, out_vals, mask=a_mask)

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Compute tensordot first
    result = torch.tensordot(a, b, dims)
    
    # Apply reciprocal square root
    return torch.rsqrt(result)

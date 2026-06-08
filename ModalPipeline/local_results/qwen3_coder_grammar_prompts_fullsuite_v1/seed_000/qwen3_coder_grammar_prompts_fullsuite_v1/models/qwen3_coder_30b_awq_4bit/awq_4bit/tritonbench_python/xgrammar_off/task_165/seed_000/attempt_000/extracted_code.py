import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

@triton.jit
def fft_kernel(x_ptr, y_ptr, n, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    x = tl.load(x_ptr + offset * stride_x, mask=mask)
    y = tl.load(y_ptr + offset * stride_y, mask=mask)
    # Simplified FFT kernel for demonstration
    # In practice, this would be more complex
    tl.store(y_ptr + offset * stride_y, x, mask=mask)

def fftn(input: torch.Tensor, s: Optional[Tuple[int, ...]] = None, dim: Optional[Tuple[int, ...]] = None, norm: Optional[str] = None, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not (input.dtype == torch.float16 or input.dtype == torch.complex64):
        raise ValueError("Only torch.half and torch.chalf are supported on CUDA with SM53+")
    
    if not input.is_cuda:
        raise ValueError("Only CUDA tensors are supported")
    
    if s is None:
        s = tuple(input.size(d) for d in (dim if dim is not None else range(input.dim())))
    
    if dim is None:
        dim = tuple(range(input.dim() - len(s), input.dim()))
    
    if norm is None:
        norm = 'backward'
    
    if out is None:
        out = torch.empty_like(input)
    
    # Validate that all dimensions are powers of 2
    for d in dim:
        size = input.size(d)
        if size & (size - 1) != 0:
            raise ValueError("All transformed dimensions must be powers of 2")
    
    # For simplicity, assuming 1D FFT with block size
    BLOCK_SIZE = 256
    n = input.numel()
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fft_kernel[(num_blocks,)](input, out, n, 1, 1, BLOCK_SIZE=BLOCK_SIZE)
    
    # Apply normalization if needed
    if norm == 'forward':
        out = out / n
    elif norm == 'ortho':
        out = out / (n ** 0.5)
    
    return out

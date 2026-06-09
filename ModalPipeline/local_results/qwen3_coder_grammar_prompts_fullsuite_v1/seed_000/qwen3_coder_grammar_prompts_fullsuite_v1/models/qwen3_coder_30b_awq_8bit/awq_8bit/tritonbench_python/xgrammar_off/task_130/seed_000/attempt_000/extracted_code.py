import torch
import triton
import triton.language as tl
import math

@triton.jit
def _rand_kernel(out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # Generate random numbers using a simple linear congruential generator
    # Using constants from the standard LCG formula
    state = offsets + 1  # Initialize with offset to avoid zero
    # Simple hash-like transformation to generate pseudo-random numbers
    state = (state * 1103515245 + 12345) & 0x7fffffff
    # Convert to float in [0, 1) range
    rand_val = state * (1.0 / 2147483648.0)
    tl.store(out_ptr + offsets, rand_val, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Handle the case where size is a single integer or a sequence
    if len(size) == 1 and not isinstance(size[0], int):
        # size is a sequence like a list or tuple
        shape = size[0]
    else:
        shape = size
    
    # Convert shape to a tuple if it's not already
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    
    # Create the output tensor
    if out is not None:
        # Use the provided output tensor
        if out.shape != shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match requested shape {shape}")
        if dtype is not None and out.dtype != dtype:
            raise ValueError(f"Output tensor dtype {out.dtype} does not match requested dtype {dtype}")
        if device is not None and out.device != device:
            raise ValueError(f"Output tensor device {out.device} does not match requested device {device}")
        if layout != torch.strided:
            raise ValueError("Only strided layout is supported")
        if pin_memory and not out.is_cpu():
            raise ValueError("pin_memory only works with CPU tensors")
    else:
        # Create a new tensor with the specified properties
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if layout != torch.strided:
            raise ValueError("Only strided layout is supported")
        out = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        if pin_memory and out.is_cpu():
            out = out.pin_memory()
    
    # Fill the tensor with random numbers
    if out.numel() > 0:
        n = out.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _rand_kernel[grid](out, n, BLOCK=block)
    
    return out

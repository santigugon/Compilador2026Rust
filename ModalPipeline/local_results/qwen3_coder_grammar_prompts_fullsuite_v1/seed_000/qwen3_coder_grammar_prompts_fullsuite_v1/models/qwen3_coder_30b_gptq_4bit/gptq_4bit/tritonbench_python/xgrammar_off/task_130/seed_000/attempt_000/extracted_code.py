import torch
import triton
import triton.language as tl
from typing import Tuple, Optional, Union

@triton.jit
def uniform_kernel(
    output_ptr,
    size,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < size
    # Generate random numbers using a simple linear congruential generator
    # This is a simplified version for demonstration
    rand_vals = tl.rand(seed + indices, 1.0)
    tl.store(output_ptr + indices, rand_vals, mask=mask)

def rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Flatten the size arguments
    shape = size if size else ()
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    # Use Triton kernel for generation
    if total_elements > 0:
        BLOCK_SIZE = 1024
        num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks,)
        
        # Create a seed based on current time and device
        import time
        seed = int(time.time() * 1000000) % (2**32)
        
        # Launch kernel
        uniform_kernel[grid](output_ptr=output.data_ptr(), size=total_elements, seed=seed, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

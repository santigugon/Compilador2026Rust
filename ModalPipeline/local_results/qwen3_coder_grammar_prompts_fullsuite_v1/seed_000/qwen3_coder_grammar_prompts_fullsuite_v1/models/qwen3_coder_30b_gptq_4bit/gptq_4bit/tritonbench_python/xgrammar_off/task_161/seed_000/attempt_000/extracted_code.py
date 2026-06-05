import torch
import triton
import triton.language as tl

@triton.jit
def autocast_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dtype,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    if dtype == tl.float16:
        output = input.to(dtype)
    elif dtype == tl.bfloat16:
        output = input.to(dtype)
    else:
        output = input
    
    tl.store(output_ptr + offsets, output, mask=mask)

def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    """
    Context manager for mixed precision training.
    
    Args:
        device_type: Type of device ('cuda')
        enabled: Whether to enable autocast
        dtype: Target data type (torch.float16, torch.bfloat16)
        cache_enabled: Whether to enable caching
    
    Returns:
        Context manager object
    """
    if device_type != "cuda":
        raise ValueError("autocast only supports CUDA devices")
    
    class AutocastContext:
        def __init__(self):
            self.enabled = enabled
            self.dtype = dtype
            self.cache_enabled = cache_enabled
            
        def __enter__(self):
            if self.enabled:
                # Set up autocast state
                torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Clean up autocast state
            torch.amp.autocast("cuda", enabled=False)
    
    return AutocastContext()

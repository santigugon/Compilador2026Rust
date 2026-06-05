import torch
import triton
import triton.language as tl

@triton.jit
def _autocast_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    enabled: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if enabled:
        input = tl.load(input_ptr + offsets, mask=mask)
        if dtype == tl.float16:
            output = input.to(tl.float16)
        elif dtype == tl.bfloat16:
            output = input.to(tl.bfloat16)
        else:
            output = input
        tl.store(output_ptr + offsets, output, mask=mask)
    else:
        input = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, input, mask=mask)

def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    """
    Context manager for mixed precision training.
    
    Args:
        device_type: Type of device ('cuda' or 'cpu')
        enabled: Whether to enable autocast
        dtype: Target data type (torch.float16, torch.bfloat16, etc.)
        cache_enabled: Whether to enable caching
    
    Returns:
        Context manager for autocast operations
    """
    
    class AutocastContext:
        def __init__(self):
            self.enabled = enabled
            self.dtype = dtype
            self.device_type = device_type
            self.cache_enabled = cache_enabled
            
        def __enter__(self):
            # Set up autocast state
            if self.device_type == "cuda":
                if self.dtype is None:
                    self._original_dtype = torch.get_autocast_gpu_dtype()
                else:
                    self._original_dtype = self.dtype
                torch.set_autocast_gpu_dtype(self._original_dtype)
                torch.autocast("cuda", enabled=self.enabled, dtype=self.dtype)
            elif self.device_type == "cpu":
                torch.autocast("cpu", enabled=self.enabled, dtype=self.dtype)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Clean up autocast state
            pass
            
        def apply_autocast(self, input_tensor):
            """Apply autocast to a tensor"""
            if not self.enabled:
                return input_tensor
            
            if self.device_type == "cuda":
                if self.dtype is None:
                    target_dtype = torch.get_autocast_gpu_dtype()
                else:
                    target_dtype = self.dtype
            else:
                target_dtype = self.dtype if self.dtype is not None else torch.float32
                
            if input_tensor.dtype != target_dtype:
                # Use Triton kernel for efficient dtype conversion
                output_tensor = torch.empty_like(input_tensor, dtype=target_dtype)
                n_elements = input_tensor.numel()
                
                if n_elements > 0:
                    grid = (triton.cdiv(n_elements, 1024),)
                    _autocast_kernel[grid](
                        input_tensor.data_ptr(),
                        output_tensor.data_ptr(),
                        n_elements,
                        target_dtype,
                        self.enabled
                    )
                return output_tensor
            return input_tensor
    
    return AutocastContext()

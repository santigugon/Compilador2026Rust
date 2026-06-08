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
        if dtype == tl.float16:
            input = tl.load(input_ptr + offsets, mask=mask).to(tl.float16)
        elif dtype == tl.bfloat16:
            input = tl.load(input_ptr + offsets, mask=mask).to(tl.bfloat16)
        else:
            input = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, input, mask=mask)
    else:
        input = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, input, mask=mask)

class AutocastContextManager:
    def __init__(self, device_type, enabled=True, dtype=None, cache_enabled=True):
        self.device_type = device_type
        self.enabled = enabled
        self.dtype = dtype
        self.cache_enabled = cache_enabled
        self._original_dtype = None
        
    def __enter__(self):
        if self.enabled and self.dtype is not None:
            if self.dtype == torch.float16:
                self._original_dtype = torch.get_autocast_gpu_dtype()
                torch.set_autocast_gpu_dtype(torch.float16)
            elif self.dtype == torch.bfloat16:
                self._original_dtype = torch.get_autocast_gpu_dtype()
                torch.set_autocast_gpu_dtype(torch.bfloat16)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.dtype is not None and self._original_dtype is not None:
            torch.set_autocast_gpu_dtype(self._original_dtype)
            
    def _apply_triton_autocast(self, input_tensor):
        if not self.enabled:
            return input_tensor
            
        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        
        if self.dtype is None:
            dtype = tl.float32
        elif self.dtype == torch.float16:
            dtype = tl.float16
        elif self.dtype == torch.bfloat16:
            dtype = tl.bfloat16
        else:
            dtype = tl.float32
            
        grid = (triton.cdiv(n_elements, 1024),)
        
        _autocast_kernel[grid](
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            n_elements,
            dtype,
            self.enabled
        )
        
        return output_tensor

def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    return AutocastContextManager(device_type, enabled, dtype, cache_enabled)

##################################################################################################################################################



import torch

def test_autocast():
    results = {}

    # Test case 1: Basic usage with 'cuda' device type
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_1'] = tensor * 2

    # Test case 2: Explicitly disabling autocast
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_2'] = tensor * 2

    # Test case 3: Using cache_enabled set to False
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_3'] = tensor * 2

    # Test case 4: Using cache_enabled set to True
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_4'] = tensor * 2

    return results

test_results = test_autocast()

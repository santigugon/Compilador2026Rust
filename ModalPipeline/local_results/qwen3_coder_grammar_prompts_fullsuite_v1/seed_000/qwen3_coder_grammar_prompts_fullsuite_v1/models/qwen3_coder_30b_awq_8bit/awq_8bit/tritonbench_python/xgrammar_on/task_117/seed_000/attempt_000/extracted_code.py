import torch
import triton
import triton.language as tl

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Handle scalar inputs
    if not torch.is_tensor(start):
        start = torch.tensor(start, dtype=torch.get_default_dtype(), device=device)
    if not torch.is_tensor(end):
        end = torch.tensor(end, dtype=torch.get_default_dtype(), device=device)
    
    # Handle device and dtype
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create output tensor
    if out is None:
        out = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    else:
        if out.shape != (steps,):
            raise ValueError(f"Output tensor must have shape ({steps},)")
        if out.dtype != dtype:
            raise ValueError(f"Output tensor must have dtype {dtype}")
        if out.device != device:
            raise ValueError(f"Output tensor must be on device {device}")
    
    # Handle scalar start and end
    start_val = start.item() if start.numel() == 1 else start
    end_val = end.item() if end.numel() == 1 else end
    
    # Special case: steps = 1
    if steps == 1:
        out.fill_(torch.pow(torch.tensor(base), torch.tensor(start_val)))
        return out
    
    # Special case: steps = 2
    if steps == 2:
        out[0] = torch.pow(torch.tensor(base), torch.tensor(start_val))
        out[1] = torch.pow(torch.tensor(base), torch.tensor(end_val))
        return out
    
    # For general case, use Triton kernel
    @triton.jit
    def _logspace_kernel(start_ptr, end_ptr, out_ptr, base_val: tl.constexpr, steps: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < steps
        
        # Compute linearly spaced values from start to end
        start_val = tl.load(start_ptr)
        end_val = tl.load(end_ptr)
        
        # Linear interpolation
        t = tl.cast(offsets, tl.float32) / tl.cast(steps - 1, tl.float32)
        log_val = start_val + t * (end_val - start_val)
        
        # Apply base^x
        result = tl.pow(base_val, log_val)
        tl.store(out_ptr + offsets, result, mask=mask)
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(steps, block),)
    
    _logspace_kernel[grid](start, end, out, base, steps, BLOCK=block)
    return out
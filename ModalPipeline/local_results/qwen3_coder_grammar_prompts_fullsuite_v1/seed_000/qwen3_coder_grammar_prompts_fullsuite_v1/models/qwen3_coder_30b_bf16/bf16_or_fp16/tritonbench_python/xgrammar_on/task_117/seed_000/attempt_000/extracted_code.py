import torch
import triton
import triton.language as tl

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Handle scalar inputs
    if not torch.is_tensor(start):
        start = torch.tensor(start, dtype=torch.get_default_dtype())
    if not torch.is_tensor(end):
        end = torch.tensor(end, dtype=torch.get_default_dtype())
    
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
    
    # Handle scalar tensors
    start_val = start.item() if start.numel() == 1 else start
    end_val = end.item() if end.numel() == 1 else end
    
    # Special case: steps = 1
    if steps == 1:
        out.fill_(torch.pow(torch.tensor(base), start_val))
        return out
    
    # Special case: steps = 2
    if steps == 2:
        out[0] = torch.pow(torch.tensor(base), start_val)
        out[1] = torch.pow(torch.tensor(base), end_val)
        return out
    
    # For general case, use Triton kernel
    @triton.jit
    def _logspace_kernel(start_val, end_val, base_val, out_ptr, steps: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < steps
        
        # Compute linearly spaced indices
        indices = tl.cast(offsets, tl.float32)
        # Scale indices from 0 to 1
        scaled_indices = indices / (steps - 1)
        # Interpolate between start and end
        log_indices = start_val + scaled_indices * (end_val - start_val)
        # Apply base^log_index
        values = tl.pow(base_val, log_indices)
        tl.store(out_ptr + offsets, values, mask=mask)
    
    block = 256
    grid = (triton.cdiv(steps, block),)
    
    # Convert scalars to tensors for Triton kernel
    start_tensor = torch.tensor(start_val, dtype=torch.get_default_dtype(), device=device)
    end_tensor = torch.tensor(end_val, dtype=torch.get_default_dtype(), device=device)
    base_tensor = torch.tensor(base, dtype=torch.get_default_dtype(), device=device)
    
    _logspace_kernel[grid](start_tensor, end_tensor, base_tensor, out, steps, BLOCK=block)
    return out
import torch
import triton
import triton.language as tl

@triton.jit
def _logspace_kernel(start_ptr, end_ptr, out_ptr, steps: tl.constexpr, base: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < steps
    
    # Load start and end values (they are scalars, so we broadcast them)
    start_val = tl.load(start_ptr)
    end_val = tl.load(end_ptr)
    
    # Compute the logarithmic spacing
    # We want: base^(start + i * (end - start) / (steps - 1)) for i in [0, steps-1]
    # This is equivalent to: base^(start) * base^(i * (end - start) / (steps - 1))
    # Which can be computed as: base^(start) * (base^(end - start))^(i / (steps - 1))
    
    # For better numerical stability, we compute it as:
    # base^(start + i * (end - start) / (steps - 1))
    # Using log space: exp(log(base) * (start + i * (end - start) / (steps - 1)))
    
    # Compute the step size
    if steps > 1:
        step_size = (end_val - start_val) / (steps - 1)
    else:
        step_size = 0.0
    
    # Compute indices
    indices = start_val + offsets * step_size
    
    # Compute base^indices
    # Using log space for numerical stability
    log_base = tl.log(base)
    log_result = log_base * indices
    result = tl.exp(log_result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Handle scalar inputs
    if not torch.is_tensor(start):
        start = torch.tensor(start, dtype=torch.get_default_dtype())
    if not torch.is_tensor(end):
        end = torch.tensor(end, dtype=torch.get_default_dtype())
    
    # Validate steps
    if steps <= 0:
        raise ValueError("steps must be positive")
    
    # Determine output tensor
    if out is not None:
        result = out
    else:
        # Create output tensor with appropriate dtype and device
        if dtype is None:
            dtype = torch.get_default_dtype()
        result = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    
    # Handle special case of 1 step
    if steps == 1:
        with torch.no_grad():
            result.fill_(torch.pow(base, start.item()))
        return result
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(steps, block),)
    
    _logspace_kernel[grid](start, end, result, steps, base, BLOCK=block)
    
    return result

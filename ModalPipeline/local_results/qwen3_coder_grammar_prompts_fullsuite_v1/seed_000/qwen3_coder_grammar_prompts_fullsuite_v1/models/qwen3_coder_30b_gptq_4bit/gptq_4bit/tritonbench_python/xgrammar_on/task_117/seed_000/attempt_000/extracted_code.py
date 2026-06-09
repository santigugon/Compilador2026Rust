import torch
import triton
import triton.language as tl

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Handle scalar inputs
    if not torch.is_tensor(start):
        start = torch.tensor(start, dtype=torch.float32)
    if not torch.is_tensor(end):
        end = torch.tensor(end, dtype=torch.float32)
    
    # Ensure start and end are 0-dimensional tensors
    if start.ndim != 0 or end.ndim != 0:
        raise ValueError("start and end must be 0-dimensional tensors")
    
    # Determine output dtype
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create output tensor
    if out is None:
        out = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    else:
        if out.shape != (steps,):
            raise ValueError("out tensor must have shape (steps,)")
        if out.dtype != dtype:
            raise ValueError("out tensor must have the correct dtype")
        if out.device != device:
            raise ValueError("out tensor must be on the correct device")
    
    # Handle the case where start == end
    if start.item() == end.item():
        out.fill_(base ** start.item())
        return out
    
    # Convert to float for computation
    start_val = start.item()
    end_val = end.item()
    
    # Create the logspace values
    # We compute: base^(start + i * (end - start) / (steps - 1)) for i in range(steps)
    # This is equivalent to: base^(start) * base^(i * (end - start) / (steps - 1))
    # Which is: base^(start) * (base^(end - start))^(i / (steps - 1))
    
    # Use Triton kernel for computation
    _logspace_kernel[(1,)](out, start_val, end_val, base, steps)
    return out

@triton.jit
def _logspace_kernel(out_ptr, start_val: tl.constexpr, end_val: tl.constexpr, base: tl.constexpr, steps: tl.constexpr):
    # Compute the logarithmic spacing
    # We want: base^(start + i * (end - start) / (steps - 1))
    # For each element i from 0 to steps-1
    
    # Create a block of indices
    indices = tl.arange(0, steps)
    
    # Compute the exponent for each index
    # exponent = start + i * (end - start) / (steps - 1)
    # But we need to be careful with division
    
    # Compute the step size
    step_size = (end_val - start_val) / (steps - 1)
    
    # Compute the exponent for each index
    exponents = start_val + indices * step_size
    
    # Compute base^exponent for each index
    # We compute: base^exponents
    # Using the identity: base^x = exp(x * log(base))
    # So: base^exponents = exp(exponents * log(base))
    
    # Compute log(base)
    log_base = tl.log(base)
    
    # Compute the values
    values = tl.exp(exponents * log_base)
    
    # Store the results
    tl.store(out_ptr + indices, values, mask=indices < steps)
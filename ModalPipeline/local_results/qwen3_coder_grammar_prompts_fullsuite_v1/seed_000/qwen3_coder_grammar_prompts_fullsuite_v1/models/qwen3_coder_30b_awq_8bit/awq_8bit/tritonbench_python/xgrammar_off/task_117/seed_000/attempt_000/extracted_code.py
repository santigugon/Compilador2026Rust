import torch
import triton
import triton.language as tl

@triton.jit
def logspace_kernel(
    start_ptr, end_ptr, out_ptr, 
    start_val, end_val, steps, base,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < steps
    
    # Generate linearly spaced indices
    indices = tl.cast(offsets, tl.float32)
    # Linear interpolation between start and end
    linear_vals = start_val + (end_val - start_val) * indices / (steps - 1)
    # Apply logarithmic transformation
    log_vals = tl.log(linear_vals) / tl.log(base)
    # Exponentiate to get final values
    result = tl.pow(base, log_vals)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    if steps <= 0:
        raise ValueError("steps must be positive")
    
    # Handle tensor inputs
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()
    
    # Determine output dtype
    if dtype is None:
        if isinstance(start, complex) or isinstance(end, complex):
            dtype = torch.complex64
        else:
            dtype = torch.get_default_dtype()
    
    # Create output tensor
    if out is None:
        out = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    else:
        if out.shape != (steps,):
            raise ValueError("out tensor must have shape (steps,)")
        if out.dtype != dtype:
            raise ValueError("out tensor must have the same dtype")
        if out.device != (device or torch.device('cpu')):
            raise ValueError("out tensor must be on the same device")
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (steps + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Convert to appropriate types for kernel
    start_val = float(start)
    end_val = float(end)
    base_val = float(base)
    
    logspace_kernel[grid_size](
        start_ptr=None, end_ptr=None, out_ptr=out.data_ptr(),
        start_val=start_val, end_val=end_val, steps=steps, base=base_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

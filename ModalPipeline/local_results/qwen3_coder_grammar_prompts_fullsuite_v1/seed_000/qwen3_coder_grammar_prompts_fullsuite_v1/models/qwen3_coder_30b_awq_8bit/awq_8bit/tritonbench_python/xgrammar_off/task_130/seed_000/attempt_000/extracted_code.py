import torch
import triton
import triton.language as tl

@triton.jit
def rand_kernel(output_ptr, n_elements, seed, offset):
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    # Simple linear congruential generator
    state = seed + offsets
    state = (state * 1103515245 + 12345) & 0x7fffffff
    rand_vals = state.astype(tl.float32) / 2147483648.0
    tl.store(output_ptr + offsets, rand_vals, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    if isinstance(size[0], (list, tuple)):
        shape = size[0]
    else:
        shape = size
    
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    if out is not None:
        output = out
    else:
        output = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    if total_elements == 0:
        return output
    
    grid = (triton.cdiv(total_elements, 1024),)
    if generator is not None:
        seed = generator.seed()
    else:
        seed = torch.seed() % (2**32)
    
    rand_kernel[grid](output.data_ptr(), total_elements, seed, 0)
    return output

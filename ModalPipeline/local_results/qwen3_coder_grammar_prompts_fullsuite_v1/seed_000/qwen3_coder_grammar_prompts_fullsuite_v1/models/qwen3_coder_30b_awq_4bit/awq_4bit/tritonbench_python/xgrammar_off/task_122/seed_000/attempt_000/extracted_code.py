import torch
import triton
import triton.language as tl

@triton.jit
def pow_kernel(
    x_ptr, 
    exp_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
    is_scalar: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    if is_scalar:
        exponent = tl.load(exp_ptr)
        result = tl.power(x, exponent)
    else:
        exp = tl.load(exp_ptr + offsets, mask=mask)
        result = tl.power(x, exp)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    if isinstance(exponent, (int, float)):
        is_scalar = True
        exp_tensor = torch.tensor(exponent, dtype=torch.float32, device=input.device)
    else:
        is_scalar = False
        exp_tensor = exponent
    
    if not is_scalar and input.shape != exp_tensor.shape:
        # Handle broadcasting
        try:
            torch.broadcast_tensors(input, exp_tensor)
        except RuntimeError:
            raise ValueError("Input and exponent tensors are not broadcastable")
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    pow_kernel[grid](
        input.data_ptr(),
        exp_tensor.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
        is_scalar
    )
    
    return out

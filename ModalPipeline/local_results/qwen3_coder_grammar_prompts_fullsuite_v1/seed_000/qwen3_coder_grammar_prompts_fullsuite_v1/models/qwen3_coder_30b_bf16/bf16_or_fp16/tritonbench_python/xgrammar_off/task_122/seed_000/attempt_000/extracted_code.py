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
    IS_SCALAR: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    if IS_SCALAR:
        exp = exp_ptr[0]
        result = tl.libdevice.pow(x, exp)
    else:
        exp = tl.load(exp_ptr + offsets, mask=mask)
        result = tl.libdevice.pow(x, exp)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if not isinstance(exponent, (torch.Tensor, float, int)):
        raise TypeError("exponent must be a torch.Tensor, float, or int")
    
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    
    if not isinstance(exponent, torch.Tensor):
        # Scalar exponent case
        is_scalar = True
        exp_tensor = torch.tensor([float(exponent)], dtype=torch.float32, device=input.device)
    else:
        # Tensor exponent case
        is_scalar = False
        exp_tensor = exponent.to(input.dtype).to(input.device)
    
    if out is None:
        out = torch.empty_like(input)
    
    if input.shape != out.shape:
        raise ValueError("output tensor must have the same shape as input tensor")
    
    if not is_scalar and input.shape != exp_tensor.shape:
        # Check if shapes are broadcastable
        try:
            torch.broadcast_tensors(input, exp_tensor)
        except RuntimeError:
            raise ValueError("input and exponent tensors must be broadcastable")
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    pow_kernel[grid](
        input.data_ptr(),
        exp_tensor.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
        is_scalar
    )
    
    return out

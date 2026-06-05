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
        result = tl.pow(x, exp)
    else:
        exp = tl.load(exp_ptr + offsets, mask=mask)
        result = tl.pow(x, exp)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def pow(input, exponent, *, out=None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if not isinstance(exponent, (torch.Tensor, float, int)):
        raise TypeError("exponent must be a torch.Tensor, float, or int")
    
    if not input.is_contiguous():
        input = input.contiguous()
    
    if isinstance(exponent, torch.Tensor):
        if not exponent.is_contiguous():
            exponent = exponent.contiguous()
        if exponent.numel() == 1:
            exponent = exponent.item()
        else:
            if input.shape != exponent.shape:
                try:
                    torch.broadcast_tensors(input, exponent)
                except RuntimeError:
                    raise ValueError("input and exponent shapes are not broadcastable")
    
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")
        if not out.is_contiguous():
            out = out.contiguous()
    else:
        out = torch.empty_like(input)
    
    if isinstance(exponent, (float, int)):
        exponent = torch.tensor(exponent, dtype=torch.float32, device=input.device)
        is_scalar = True
    else:
        is_scalar = False
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    pow_kernel[grid](
        input.data_ptr(),
        exponent.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
        is_scalar
    )
    
    return out

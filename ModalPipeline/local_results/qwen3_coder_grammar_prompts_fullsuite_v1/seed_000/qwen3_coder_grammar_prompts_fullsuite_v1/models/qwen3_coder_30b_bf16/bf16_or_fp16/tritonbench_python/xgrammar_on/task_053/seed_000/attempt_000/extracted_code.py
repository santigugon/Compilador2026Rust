import torch
import triton
import triton.language as tl

@triton.jit
def _mul_relu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, inplace: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x * y
    # Apply ReLU
    relu_result = tl.maximum(result, 0.0)
    if inplace:
        tl.store(x_ptr + offsets, relu_result, mask=mask)
    else:
        tl.store(out_ptr + offsets, relu_result, mask=mask)


def mul_relu(input, other, inplace=False, out=None) -> torch.Tensor:
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure tensors have the same device and dtype
    other = other.to(input.device, input.dtype)
    
    # Handle broadcasting
    input, other = torch.broadcast_tensors(input, other)
    
    if inplace:
        if out is not None:
            raise ValueError("Cannot specify both 'inplace=True' and 'out' parameter.")
        out = input
        # Check if input is contiguous for in-place operation
        if not input.is_contiguous():
            input = input.contiguous()
    else:
        if out is None:
            out = torch.empty_like(input)
        else:
            if out.shape != input.shape:
                raise ValueError("Output tensor shape does not match input shape.")
            if out.dtype != input.dtype or out.device != input.device:
                raise ValueError("Output tensor must have the same dtype and device as input.")
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # If other is a scalar, we need to handle it specially
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    _mul_relu_kernel[grid](input, other, out, n, inplace, BLOCK=block)
    return out
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
    # Apply ReLU: max(0, result)
    relu_result = tl.maximum(0.0, result)
    if inplace:
        tl.store(x_ptr + offsets, relu_result, mask=mask)
    else:
        tl.store(out_ptr + offsets, relu_result, mask=mask)

def mul_relu(input, other, inplace=False, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other is broadcastable with input
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Determine output tensor
    if inplace:
        if out is not None:
            raise ValueError("Cannot specify both 'inplace=True' and 'out'")
        out = input
    elif out is None:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle the case where we're doing in-place operation
    if inplace:
        _mul_relu_kernel[grid](input, other, input, n, True, BLOCK=block)
    else:
        _mul_relu_kernel[grid](input, other, out, n, False, BLOCK=block)
    
    return out

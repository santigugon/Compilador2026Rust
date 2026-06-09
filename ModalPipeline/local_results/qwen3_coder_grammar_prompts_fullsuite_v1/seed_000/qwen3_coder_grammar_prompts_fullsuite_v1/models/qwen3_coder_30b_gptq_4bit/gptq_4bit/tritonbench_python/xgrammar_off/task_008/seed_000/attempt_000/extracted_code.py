import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x - alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure both tensors have the same device
    if other.device != input.device:
        other = other.to(input.device)
    
    # Handle broadcasting by creating a unified shape
    # For simplicity, we'll use torch's built-in broadcasting
    # and just ensure the output tensor is correctly sized
    out_shape = torch.broadcast_tensors(input, other)[0].shape
    
    # Create output tensor
    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    else:
        # Ensure output tensor has the correct shape and dtype
        if out.shape != out_shape or out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same shape and dtype as the broadcasted result")
    
    # Flatten tensors for kernel execution
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle the case where other is a scalar
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same shape as input for element-wise operations
    other = other.expand_as(input)
    
    # Launch kernel
    _sub_kernel[grid](input, other, out, n, alpha, BLOCK=block)
    return out

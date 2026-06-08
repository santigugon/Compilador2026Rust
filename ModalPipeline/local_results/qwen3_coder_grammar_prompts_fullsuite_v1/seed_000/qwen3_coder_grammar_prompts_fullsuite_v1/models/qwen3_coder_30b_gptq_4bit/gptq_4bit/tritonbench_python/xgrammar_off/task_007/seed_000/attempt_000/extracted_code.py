import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _argmax_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For simplicity, we'll compute argmax in a straightforward way
    # This is a simplified version - a full implementation would require
    # more complex logic to handle the reduction properly
    max_val = tl.full([BLOCK], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK], 0, dtype=tl.int32)
    
    # This is a simplified approach - in practice, a more complex reduction
    # kernel would be needed for proper argmax computation
    for i in range(dim_size):
        current_val = tl.load(x_ptr + i * BLOCK + tl.arange(0, BLOCK), mask=mask, other=0.0)
        mask_update = current_val > max_val
        max_val = tl.where(mask_update, current_val, max_val)
        max_idx = tl.where(mask_update, i, max_idx)
    
    tl.store(out_ptr + offsets, max_idx, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
    
    # Apply sigmoid
    input_sigmoid = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sigmoid_kernel[grid](input, input_sigmoid, n, BLOCK=block)
    
    # Compute argmax
    if dim is None:
        # Flatten the tensor and find argmax
        flat_tensor = input_sigmoid.flatten()
        argmax_idx = torch.argmax(flat_tensor)
        if keepdim:
            return argmax_idx.unsqueeze(0)
        else:
            return argmax_idx
    else:
        # Find argmax along specified dimension
        argmax_idx = torch.argmax(input_sigmoid, dim=dim, keepdim=keepdim)
        return argmax_idx

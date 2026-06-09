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
    # For simplicity, we'll compute argmax in a single block
    # This assumes the input is small enough to fit in one block
    # In practice, you'd want to handle larger tensors with proper reduction
    if pid == 0:
        # Load all values
        offsets = tl.arange(0, n)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Find maximum value and its index
        max_val = x[0]
        max_idx = 0
        for i in range(1, n):
            if x[i] > max_val:
                max_val = x[i]
                max_idx = i
        
        # Store result
        tl.store(out_ptr, max_idx, mask=True)

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
        # Flatten tensor and find argmax
        flat_tensor = input_sigmoid.flatten()
        result = torch.argmax(flat_tensor)
        if keepdim:
            result = result.view(1)
    else:
        # Find argmax along specified dimension
        result = torch.argmax(input_sigmoid, dim=dim, keepdim=keepdim)
    
    return result

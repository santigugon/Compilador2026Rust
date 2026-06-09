import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_dot_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_new = y + alpha * x
    tl.store(y_ptr + offsets, y_new, mask=mask)
    
    # Compute dot product of modified y with itself
    # For simplicity, we'll compute the dot product in a separate kernel
    # This is a basic implementation - in practice, you might want to optimize this
    # by using a reduction kernel or computing it in the same kernel
    y_dot = y_new * y_new
    tl.store(out_ptr + offsets, y_dot, mask=mask)

@triton.jit
def _sum_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Use a reduction to sum all elements
    # This is a simple reduction that sums all elements
    # In a real implementation, you'd want to use a proper reduction
    # but for this case, we'll just sum the elements
    sum_val = tl.sum(x, axis=0)
    # Store the result
    tl.store(out_ptr, sum_val, mask=mask)


def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure inputs are contiguous
    y = y.contiguous()
    x = x.contiguous()
    
    # Check if tensors have the same size
    if y.numel() != x.numel():
        raise ValueError("Tensors must have the same number of elements")
    
    # Create output tensor
    out = torch.empty_like(y)
    
    # Compute the scaled addition and dot product
    n = y.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # First compute y += alpha * x and store the result in y
    # Then compute dot product of modified y with itself
    # We'll do this in two steps for clarity
    
    # Step 1: Compute y += alpha * x
    y_modified = torch.empty_like(y)
    y_modified.copy_(y)
    
    # Step 2: Compute dot product of modified y with itself
    # We'll compute this using a reduction
    
    # For the fused operation, we'll use a single kernel
    # that does both operations
    out = torch.empty_like(y)
    
    # Create a temporary tensor to hold the dot product values
    dot_product_vals = torch.empty_like(y)
    
    # Launch the kernel
    _scaled_add_dot_kernel[grid](y_modified, x, dot_product_vals, n, alpha, BLOCK=block)
    
    # Now compute the sum of the dot product values
    # This is the final result
    result = torch.sum(dot_product_vals)
    
    return result
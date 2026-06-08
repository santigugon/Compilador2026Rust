import torch
import triton
import triton.language as tl

@triton.jit
def chebyshev_kernel(input_ptr, n_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    input = tl.load(input_ptr + offsets, mask=mask)
    n_val = tl.load(n_ptr)
    
    # Handle special cases
    # If n = 0, return 1
    # If n = 1, return input
    # For n < 6 or |input| > 1, use recursive formula
    # Otherwise, use trigonometric formula
    
    # Initialize output tensor
    output = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)
    
    # Handle n = 0 case
    if n_val == 0:
        output = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    # Handle n = 1 case
    elif n_val == 1:
        output = input
    # Handle n < 6 or |input| > 1 case (recursive)
    elif n_val < 6 or tl.abs(input) > 1:
        # Recursive formula: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        # Base cases
        if n_val == 1:
            output = input
        else:
            # Initialize T_0 = 1, T_1 = x
            t_prev2 = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
            t_prev1 = input
            # Compute recursively
            for i in range(2, n_val + 1):
                t_current = 2.0 * input * t_prev1 - t_prev2
                t_prev2 = t_prev1
                t_prev1 = t_current
            output = t_prev1
    else:
        # Use trigonometric formula for large n and |input| <= 1
        # T_n(x) = cos(n * arccos(x)) for |x| <= 1
        # This is a simplified version - in practice, you'd want to handle
        # the case where x is exactly 1 or -1 more carefully
        arccos_x = tl.acos(input)
        output = tl.cos(n_val * arccos_x)
    
    tl.store(out_ptr + offsets, output, mask=mask)

def chebyshev_polynomial_t(input, n, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and n are on the same device
    device = input.device
    n = n.to(device)
    
    # Ensure input and out are contiguous
    input = input.contiguous()
    out = out.contiguous()
    
    # Launch kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create a scalar tensor for n to pass to kernel
    n_scalar = torch.tensor(n.item(), dtype=torch.int32, device=device)
    
    chebyshev_kernel[grid](
        input,
        n_scalar,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

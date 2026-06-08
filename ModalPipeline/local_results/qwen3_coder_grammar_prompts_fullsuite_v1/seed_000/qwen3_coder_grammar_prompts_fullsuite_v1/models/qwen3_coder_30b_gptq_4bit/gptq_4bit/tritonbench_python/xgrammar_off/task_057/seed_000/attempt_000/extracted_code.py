import torch
import triton
import triton.language as tl

@triton.jit
def _chebyshev_polynomial_t_kernel(
    input_ptr, 
    out_ptr, 
    n, 
    size: tl.constexpr, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # For n = 0, return 1
    # For n = 1, return input
    # For n < 6 or |input| > 1, use recursive formula
    # Otherwise, use trigonometric formula
    
    # Initialize output
    result = tl.zeros_like(input_val)
    
    # Handle n = 0 case
    if n == 0:
        result = tl.full_like(input_val, 1.0)
    # Handle n = 1 case
    elif n == 1:
        result = input_val
    # For n < 6 or |input| > 1, use recursive formula
    elif n < 6 or tl.abs(input_val) > 1:
        # Recursive formula: T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        # We compute T_n(x) iteratively
        # T_0(x) = 1
        # T_1(x) = x
        # T_2(x) = 2 * x^2 - 1
        # T_3(x) = 4 * x^3 - 3 * x
        # etc.
        
        # Initialize T_0 and T_1
        T_prev2 = tl.full_like(input_val, 1.0)  # T_0
        T_prev1 = input_val  # T_1
        
        # For n = 2, T_2 = 2 * x^2 - 1
        if n == 2:
            result = 2.0 * input_val * input_val - 1.0
        # For n = 3, T_3 = 4 * x^3 - 3 * x
        elif n == 3:
            result = 4.0 * input_val * input_val * input_val - 3.0 * input_val
        # For n = 4, T_4 = 8 * x^4 - 8 * x^2 + 1
        elif n == 4:
            result = 8.0 * input_val * input_val * input_val * input_val - 8.0 * input_val * input_val + 1.0
        # For n = 5, T_5 = 16 * x^5 - 20 * x^3 + 5 * x
        elif n == 5:
            result = 16.0 * input_val * input_val * input_val * input_val * input_val - 20.0 * input_val * input_val * input_val + 5.0 * input_val
        # For n >= 6, use recursive formula
        else:
            # Compute T_n using recurrence relation
            for i in range(2, n + 1):
                T_current = 2.0 * input_val * T_prev1 - T_prev2
                T_prev2 = T_prev1
                T_prev1 = T_current
            result = T_prev1
    
    # For n >= 6 and |input| <= 1, use trigonometric formula
    else:
        # T_n(x) = cos(n * arccos(x))
        # We compute arccos(x) first
        arccos_x = tl.acos(input_val)
        # Then compute cos(n * arccos(x))
        result = tl.cos(n * arccos_x)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def chebyshev_polynomial_t(input, n, *, out=None):
    # Handle scalar n
    if not torch.is_tensor(n):
        n_scalar = n
        n = torch.tensor(n_scalar, dtype=torch.int32, device=input.device)
    else:
        n_scalar = n.item()
    
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Get size
    size = input.numel()
    
    # Determine block size
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    
    # Launch kernel
    _chebyshev_polynomial_t_kernel[grid](
        input, 
        out, 
        n_scalar, 
        size, 
        BLOCK=BLOCK
    )
    
    # If input was scalar, squeeze output
    if input.shape == (1,):
        out = out.squeeze(0)
    
    return out

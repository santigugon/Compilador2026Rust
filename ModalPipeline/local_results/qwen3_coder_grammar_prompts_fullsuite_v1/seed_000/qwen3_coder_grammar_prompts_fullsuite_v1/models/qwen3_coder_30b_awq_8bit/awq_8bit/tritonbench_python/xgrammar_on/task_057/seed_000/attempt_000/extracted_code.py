import torch
import triton
import triton.language as tl

@triton.jit
def _chebyshev_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For n = 0, T_0(x) = 1
    # For n = 1, T_1(x) = x
    # For n >= 2, use recurrence relation or trigonometric formula
    
    # Initialize T_0 and T_1
    t0 = 1.0
    t1 = x
    
    # For n = 0, return 1
    result = tl.where(n == 0, 1.0, t1)
    
    # For n = 1, return x
    # For n >= 2, compute recursively
    for i in range(2, n + 1):
        # T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        t2 = 2.0 * x * t1 - t0
        t0 = t1
        t1 = t2
        result = tl.where(n == i, t1, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _chebyshev_trig_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Use trigonometric formula for |x| <= 1 and n >= 6
    # T_n(x) = cos(n * acos(x))
    # But we need to handle special cases
    # For |x| > 1, we use recurrence relation
    
    # Check if |x| <= 1
    x_abs = tl.abs(x)
    
    # For |x| <= 1 and n >= 6
    # T_n(x) = cos(n * acos(x))
    # But we need to be careful with domain of acos
    # acos(x) is defined for x in [-1, 1]
    
    # For |x| <= 1
    acos_x = tl.where(x_abs <= 1.0, tl.acos(x), 0.0)
    # For |x| > 1, we use recurrence relation
    
    # Initialize T_0 and T_1
    t0 = 1.0
    t1 = x
    
    # For n = 0, return 1
    result = tl.where(n == 0, 1.0, t1)
    
    # For n = 1, return x
    # For n >= 2, compute recursively
    for i in range(2, n + 1):
        # T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        t2 = 2.0 * x * t1 - t0
        t0 = t1
        t1 = t2
        result = tl.where(n == i, t1, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _chebyshev_trig_kernel_optimized(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For |x| <= 1 and n >= 6, use trigonometric formula
    # T_n(x) = cos(n * acos(x))
    # But we need to handle special cases
    
    # For |x| <= 1
    x_abs = tl.abs(x)
    
    # For n = 0, return 1
    result = tl.where(n == 0, 1.0, x)
    
    # For n = 1, return x
    # For n >= 2, compute recursively
    for i in range(2, n + 1):
        # T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        # But for n >= 6 and |x| <= 1, we can use trigonometric formula
        # T_n(x) = cos(n * acos(x))
        # However, we'll stick to recurrence relation for simplicity
        # and to avoid domain issues with acos
        
        # Initialize T_0 and T_1
        t0 = 1.0
        t1 = x
        
        # Compute T_i for i = 2 to n
        for j in range(2, i + 1):
            t2 = 2.0 * x * t1 - t0
            t0 = t1
            t1 = t2
        
        result = tl.where(n == i, t1, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _chebyshev_kernel_optimized(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For n = 0, T_0(x) = 1
    # For n = 1, T_1(x) = x
    # For n >= 2, use recurrence relation
    
    # Initialize T_0 and T_1
    t0 = 1.0
    t1 = x
    
    # For n = 0, return 1
    result = tl.where(n == 0, 1.0, t1)
    
    # For n = 1, return x
    # For n >= 2, compute recursively
    for i in range(2, n + 1):
        # T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        t2 = 2.0 * x * t1 - t0
        t0 = t1
        t1 = t2
        result = tl.where(n == i, t1, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def chebyshev_polynomial_t(input, n, *, out=None):
    # Convert n to a scalar if it's a tensor
    if torch.is_tensor(n):
        n = n.item()
    
    # Handle special cases
    if n == 0:
        if out is not None:
            out.fill_(1.0)
            return out
        return torch.ones_like(input)
    
    if n == 1:
        if out is not None:
            out.copy_(input)
            return out
        return input.clone()
    
    # For n >= 2, use recurrence relation
    # This is a simplified version that always uses recurrence relation
    # For better performance, we could add trigonometric formula for |x| <= 1 and n >= 6
    
    if out is not None:
        out_tensor = out
    else:
        out_tensor = torch.empty_like(input)
    
    size = input.numel()
    block = 256
    grid = (triton.cdiv(size, block),)
    
    _chebyshev_kernel_optimized[grid](input, out_tensor, n, size, BLOCK=block)
    return out_tensor
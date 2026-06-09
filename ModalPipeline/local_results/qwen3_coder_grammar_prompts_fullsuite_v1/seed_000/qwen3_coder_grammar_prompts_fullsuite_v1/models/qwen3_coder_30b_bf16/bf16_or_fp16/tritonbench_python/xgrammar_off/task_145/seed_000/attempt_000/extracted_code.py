import torch
import triton
import triton.language as tl

@triton.jit
def _polygamma_kernel(n, input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute polygamma function using recurrence relation
    # For n=0, polygamma is digamma function
    # For n>0, polygamma(n,x) = (-1)^(n+1) * n! * sum(1/(x+k)^(n+1))
    
    # Initialize result
    result = tl.zeros_like(input)
    
    # Special case for n=0 (digamma)
    if n == 0:
        # Simple approximation for digamma function
        # Using asymptotic expansion: digamma(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ...
        x = input
        result = tl.log(x) - 0.5 / x
        x_sq = x * x
        result = result - 1.0 / (12.0 * x_sq)
        x_quad = x_sq * x_sq
        result = result + 1.0 / (120.0 * x_quad)
    else:
        # For higher order polygamma functions
        # Using the recurrence relation: psi^(n)(x) = (-1)^(n+1) * n! * sum(1/(x+k)^(n+1))
        # We'll compute a few terms of the series
        result = tl.zeros_like(input)
        sign = tl.where(n % 2 == 0, 1.0, -1.0)
        factorial_n = 1.0
        for i in range(1, n + 1):
            factorial_n *= i
        
        # Compute sum term by term
        # We'll use a simple approximation with a fixed number of terms
        for k in range(100):
            term = 1.0 / tl.pow(input + k, n + 1)
            result = result + term
            # Early stopping condition
            if tl.abs(term) < 1e-12:
                break
        
        result = sign * factorial_n * result
    
    tl.store(output_ptr + offsets, result, mask=mask)

def polygamma(n, input, *, out=None) -> torch.Tensor:
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if input.is_cuda:
        # Create output tensor if not provided
        if out is None:
            out = torch.empty_like(input, dtype=torch.float32)
        else:
            if out.shape != input.shape:
                raise ValueError("out tensor must have the same shape as input")
            if out.dtype != torch.float32:
                raise ValueError("out tensor must have dtype torch.float32")
        
        # Launch kernel
        grid = (triton.cdiv(input.numel(), 1024),)
        _polygamma_kernel[grid](
            n,
            input.data_ptr(),
            out.data_ptr(),
            input.numel(),
            BLOCK_SIZE=1024
        )
        return out
    else:
        # Fall back to CPU implementation
        if n == 0:
            # Use torch.digamma for digamma function
            result = torch.digamma(input)
        else:
            # For higher order polygamma, we'll use a simple approximation
            # This is a placeholder implementation
            result = torch.zeros_like(input, dtype=torch.float32)
            # This is a simplified version - a full implementation would be more complex
            x = input
            # Simple approximation for polygamma
            result = torch.pow(x, -n - 1) * (-1)**(n + 1) * torch.lgamma(n + 1)
        
        if out is not None:
            out.copy_(result)
            return out
        return result

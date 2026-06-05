import torch
import triton
import triton.language as tl

@triton.jit
def digamma_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # For numerical stability, we use the asymptotic expansion
    # digamma(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    # We compute this using a series expansion for better accuracy
    
    # Handle special case where input is 0
    output = tl.where(input == 0, -float('inf'), tl.zeros_like(input))
    
    # For x > 0, use the asymptotic expansion
    x = input
    x_sq = x * x
    x_pow_2 = x_sq
    x_pow_4 = x_sq * x_sq
    x_pow_6 = x_pow_2 * x_pow_4
    
    # Series expansion: ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    term1 = tl.log(x)
    term2 = 1.0 / (2.0 * x)
    term3 = 1.0 / (12.0 * x_pow_2)
    term4 = 1.0 / (120.0 * x_pow_4)
    term5 = 1.0 / (252.0 * x_pow_6)
    
    # Alternating signs
    result = term1 - term2 - term3 + term4 - term5
    
    # For small x, we use a more accurate series expansion
    # This is a simplified version for demonstration
    small_x_mask = x < 1.0
    small_x = tl.where(small_x_mask, x, 1.0)
    
    # Use the recurrence relation: digamma(x+1) = digamma(x) + 1/x
    # For small x, we compute digamma(x) = digamma(x+1) - 1/x
    # This is a simplified approximation
    result = tl.where(small_x_mask, result - 1.0 / small_x, result)
    
    # Apply the result to output
    tl.store(output_ptr + offsets, result, mask=mask)

def digamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input is on the same device as output
    input = input.to(out.device)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the kernel
    grid = (triton.cdiv(n_elements, 1024),)
    digamma_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024
    )
    
    return out

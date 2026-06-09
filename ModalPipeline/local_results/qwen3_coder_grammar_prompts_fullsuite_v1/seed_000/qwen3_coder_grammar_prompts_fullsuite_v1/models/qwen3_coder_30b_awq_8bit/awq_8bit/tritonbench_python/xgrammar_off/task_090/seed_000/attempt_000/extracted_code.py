import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_hardshrink_dropout_kernel(
    input_ptr, 
    output_ptr, 
    n: tl.constexpr,
    p: tl.constexpr,
    training: tl.constexpr,
    lambd: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout if training is True
    if training:
        # Generate random numbers for dropout
        # Using a simple hash-based approach for reproducibility
        rand_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # Simple hash to generate pseudo-random numbers
        rand_val = (rand_val * 1103515245 + 12345) & 0x7fffffff
        # Convert to probability (0-1 range)
        rand_prob = rand_val / 2147483647.0
        # Apply dropout mask
        dropout_mask = rand_prob > p
        x = tl.where(dropout_mask, x, 0.0)
    
    # Apply hard shrinkage
    # Hard shrinkage: if |x| <= lambd, return 0; otherwise return x
    abs_x = tl.abs(x)
    shrink_mask = abs_x > lambd
    result = tl.where(shrink_mask, x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor:
    # Handle scalar inputs
    if input.dim() == 0:
        # For scalar tensors, we need to handle differently
        if training:
            # Generate random number for dropout
            rand_val = torch.rand((), device=input.device, dtype=torch.float32)
            if rand_val > p:
                x = input
            else:
                x = torch.tensor(0.0, device=input.device, dtype=input.dtype)
        else:
            x = input
            
        # Apply hard shrinkage
        if torch.abs(x) <= lambd:
            result = torch.tensor(0.0, device=input.device, dtype=input.dtype)
        else:
            result = x
        return result
    
    # For non-scalar tensors
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle inplace operation
    if inplace and training:
        # For inplace operation, we need to work on the input tensor directly
        # But we need to be careful about the order of operations
        _fused_hardshrink_dropout_kernel[grid](input, input, n, p, training, lambd, BLOCK=block)
        return input
    else:
        # Normal case: use output tensor
        _fused_hardshrink_dropout_kernel[grid](input, out, n, p, training, lambd, BLOCK=block)
        return out

import torch
import triton
import triton.language as tl

def _softmax_mul_kernel(input_ptr, other_ptr, out_ptr, n_elements: tl.constexpr, dim_size: tl.constexpr, stride_input: tl.constexpr, stride_other: tl.constexpr, stride_out: tl.constexpr, BLOCK: tl.constexpr, is_other_scalar: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    
    # Load input
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply softmax
    # For softmax, we need to compute max and sum along the specified dimension
    # Since we're doing element-wise operations, we'll compute softmax per row
    # This is a simplified version assuming we're working with a flattened view
    # and the softmax is applied along the specified dimension
    
    # Compute max for numerical stability
    max_val = tl.max(input, axis=0)
    # Subtract max for numerical stability
    input_shifted = input - max_val
    # Compute exp
    exp_input = tl.exp(input_shifted)
    # Compute sum
    sum_exp = tl.sum(exp_input, axis=0)
    # Compute softmax
    softmax = exp_input / sum_exp
    
    # Load other
    if is_other_scalar:
        other = other_ptr[0]
    else:
        other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply softmax with other
    result = softmax * other
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _softmax_kernel(input_ptr, out_ptr, n_elements: tl.constexpr, dim_size: tl.constexpr, stride_input: tl.constexpr, stride_out: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    
    # Load input
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute max for numerical stability
    max_val = tl.max(input, axis=0)
    # Subtract max for numerical stability
    input_shifted = input - max_val
    # Compute exp
    exp_input = tl.exp(input_shifted)
    # Compute sum
    sum_exp = tl.sum(exp_input, axis=0)
    # Compute softmax
    softmax = exp_input / sum_exp
    
    # Store result
    tl.store(out_ptr + offsets, softmax, mask=mask)

@triton.jit
def _mul_kernel(input_ptr, other_ptr, out_ptr, n_elements: tl.constexpr, stride_input: tl.constexpr, stride_other: tl.constexpr, stride_out: tl.constexpr, BLOCK: tl.constexpr, is_other_scalar: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    
    # Load input
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load other
    if is_other_scalar:
        other = other_ptr[0]
    else:
        other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    result = input * other
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def softmax_mul(input, other, dim, dtype=None, out=None):
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
        if torch.is_tensor(other):
            other = other.to(dtype)
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Handle scalar other
    is_other_scalar = not torch.is_tensor(other)
    
    # Flatten input and other for processing
    input_flat = input.flatten()
    if is_other_scalar:
        other_flat = other
    else:
        other_flat = other.flatten()
    
    # Get number of elements
    n_elements = input_flat.numel()
    
    # Get dimension size
    dim_size = input.size(dim)
    
    # Set block size
    BLOCK = 256
    grid = (triton.cdiv(n_elements, BLOCK),)
    
    # If other is scalar, we need to handle it differently
    if is_other_scalar:
        # For scalar, we can use a simpler approach
        # First compute softmax
        softmax_out = torch.empty_like(input)
        _softmax_kernel[grid](input_flat, softmax_out.flatten(), n_elements, dim_size, input.stride(0), softmax_out.stride(0), BLOCK=BLOCK)
        # Then multiply with scalar
        _mul_kernel[grid](softmax_out.flatten(), torch.tensor([other], dtype=input.dtype, device=input.device), out.flatten(), n_elements, softmax_out.stride(0), 0, out.stride(0), BLOCK=BLOCK, is_other_scalar=True)
    else:
        # For tensor other, we need to broadcast properly
        # First compute softmax
        softmax_out = torch.empty_like(input)
        _softmax_kernel[grid](input_flat, softmax_out.flatten(), n_elements, dim_size, input.stride(0), softmax_out.stride(0), BLOCK=BLOCK)
        # Then multiply with other
        _mul_kernel[grid](softmax_out.flatten(), other_flat, out.flatten(), n_elements, softmax_out.stride(0), other.stride(0), out.stride(0), BLOCK=BLOCK, is_other_scalar=False)
    
    return out
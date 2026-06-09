import torch
import triton
import triton.language as tl

def _softplus_linear_kernel(input_ptr, weight_ptr, bias_ptr, out_ptr, n_input, n_output, beta, threshold, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program processes one output dimension
    output_idx = pid
    
    if output_idx >= n_output:
        return
    
    # Load bias if available
    if bias_ptr is not None:
        out_val = tl.load(bias_ptr + output_idx)
    else:
        out_val = 0.0
    
    # Compute linear transformation
    for i in range(0, n_input, BLOCK):
        input_offsets = i + tl.arange(0, BLOCK)
        weight_offsets = output_idx * n_input + input_offsets
        
        input_mask = input_offsets < n_input
        
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=input_mask, other=0.0)
        
        out_val += tl.sum(input_vals * weight_vals)
    
    # Apply softplus activation
    # Softplus = log(1 + exp(beta * x)) / beta
    # For numerical stability, when beta * x > threshold, we use x instead of softplus
    beta_x = beta * out_val
    
    # Use linear approximation for large values
    softplus_val = tl.where(beta_x > threshold, out_val, tl.log(1.0 + tl.exp(beta_x)) / beta)
    
    tl.store(out_ptr + output_idx, softplus_val)


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    # Validate inputs
    assert input.dim() == 1, "Input must be 1D"
    assert weight.dim() == 2, "Weight must be 2D"
    assert weight.size(1) == input.size(0), "Weight input size must match input size"
    
    n_input = input.size(0)
    n_output = weight.size(0)
    
    # Create output tensor
    out = torch.empty(n_output, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n_output, 1),)
    
    _softplus_linear_kernel[grid](
        input, weight, bias, out,
        n_input, n_output,
        beta, threshold,
        BLOCK=block
    )
    
    return out
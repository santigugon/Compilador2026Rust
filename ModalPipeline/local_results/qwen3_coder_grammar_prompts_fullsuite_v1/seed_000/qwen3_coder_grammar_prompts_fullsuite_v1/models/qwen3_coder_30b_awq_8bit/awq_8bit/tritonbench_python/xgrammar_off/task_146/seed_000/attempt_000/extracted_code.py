import torch
import triton
import triton.language as tl

@triton.jit
def _elu_linear_kernel(input_ptr, weight_ptr, bias_ptr, out_ptr, n_input, n_output, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK
    mask = offset + tl.arange(0, BLOCK) < n_input
    
    # Load input
    input = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Apply linear transformation: output = input @ weight.T + bias
    # For simplicity, we'll compute one output element at a time
    # This is a simplified approach - in practice, you'd want to use
    # proper matrix multiplication kernels for better performance
    
    # Compute linear transformation for each output dimension
    for i in range(n_output):
        # Load weight column and bias
        weight_col = tl.load(weight_ptr + i * n_input + offset, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + i, other=0.0) if bias_ptr is not None else 0.0
        
        # Compute dot product
        if i == 0:
            linear_out = input * weight_col
        else:
            linear_out += input * weight_col
            
        # Add bias
        if i == 0:
            linear_out += bias_val
        else:
            linear_out += bias_val
    
    # Apply ELU activation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    elu_out = tl.where(linear_out > 0, linear_out, alpha * (tl.exp(linear_out) - 1.0))
    
    # Store result
    tl.store(out_ptr + offset, elu_out, mask=mask)

@triton.jit
def _elu_linear_kernel_optimized(input_ptr, weight_ptr, bias_ptr, out_ptr, n_input, n_output, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK
    mask = offset + tl.arange(0, BLOCK) < n_input
    
    # Load input
    input = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Initialize output
    linear_out = tl.zeros_like(input)
    
    # Compute linear transformation: output = input @ weight.T + bias
    for i in range(n_output):
        # Load weight column and bias
        weight_col = tl.load(weight_ptr + i * n_input + offset, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + i, other=0.0) if bias_ptr is not None else 0.0
        
        # Accumulate linear transformation
        linear_out += input * weight_col
        if i == 0:
            linear_out += bias_val
    
    # Apply ELU activation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    elu_out = tl.where(linear_out > 0, linear_out, alpha * (tl.exp(linear_out) - 1.0))
    
    # Store result
    tl.store(out_ptr + offset, elu_out, mask=mask)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Handle the case where input is a scalar
    if input.dim() == 0:
        input = input.unsqueeze(0)
    
    # Get dimensions
    n_input = input.size(-1)
    n_output = weight.size(0)
    
    # Create output tensor
    if inplace:
        out = input
    else:
        out = torch.empty(input.size()[:-1] + (n_output,), dtype=input.dtype, device=input.device)
    
    # Handle bias
    if bias is not None:
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = None
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n_input, block),)
    
    # For simplicity, we'll use a basic approach for the linear transformation
    # In a real implementation, you'd want to use proper matrix multiplication
    
    # Compute linear transformation + ELU in one kernel
    _elu_linear_kernel_optimized[grid](
        input.data_ptr(), 
        weight.data_ptr(), 
        bias_ptr, 
        out.data_ptr(), 
        n_input, 
        n_output, 
        alpha, 
        BLOCK=block
    )
    
    return out

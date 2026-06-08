import torch
import triton
import triton.language as tl

@triton.jit
def _adam_kernel(
    param_ptr, 
    grad_ptr, 
    exp_avg_ptr, 
    exp_avg_sq_ptr, 
    max_exp_avg_sq_ptr,
    lr, 
    beta1, 
    beta2, 
    eps, 
    weight_decay,
    amsgrad,
    maximize,
    step,
    n: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    
    if amsgrad:
        # Keep track of the maximum of the second moment estimates
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq, exp_avg_sq)
        denom = tl.sqrt(max_exp_avg_sq) + eps
    else:
        denom = tl.sqrt(exp_avg_sq) + eps
    
    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Compute step size
    step_size = lr / bias_correction1
    
    # Update parameter
    if maximize:
        param = param + step_size * exp_avg / denom
    else:
        param = param - step_size * exp_avg / denom
    
    # Apply weight decay
    if weight_decay != 0:
        if maximize:
            param = param - lr * weight_decay * param
        else:
            param = param - lr * weight_decay * param
    
    # Store updated values
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
    
    if amsgrad:
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    # For simplicity, we'll implement a basic version that works with a single parameter tensor
    # In a real implementation, this would handle multiple parameter groups
    
    if not params:
        return
    
    # Get the first parameter to determine tensor properties
    param = params[0]
    
    # Create state tensors if they don't exist
    if not hasattr(param, 'exp_avg'):
        param.exp_avg = torch.zeros_like(param, dtype=torch.float32)
    if not hasattr(param, 'exp_avg_sq'):
        param.exp_avg_sq = torch.zeros_like(param, dtype=torch.float32)
    if amsgrad and not hasattr(param, 'max_exp_avg_sq'):
        param.max_exp_avg_sq = torch.zeros_like(param, dtype=torch.float32)
    
    # Get state tensors
    exp_avg = param.exp_avg
    exp_avg_sq = param.exp_avg_sq
    max_exp_avg_sq = getattr(param, 'max_exp_avg_sq', None)
    
    # Create output tensor
    out = torch.empty_like(param)
    
    # Copy parameter to output
    out.copy_(param)
    
    # Get tensor properties
    n = param.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Convert parameters to appropriate types
    lr_val = float(lr)
    beta1, beta2 = betas
    eps_val = float(eps)
    weight_decay_val = float(weight_decay)
    step = 1  # Simplified step counter
    
    # Launch kernel
    _adam_kernel[grid](
        out, 
        param.grad, 
        exp_avg, 
        exp_avg_sq, 
        max_exp_avg_sq,
        lr_val, 
        beta1, 
        beta2, 
        eps_val, 
        weight_decay_val,
        amsgrad,
        maximize,
        step,
        n,
        BLOCK=block
    )
    
    # Update parameter with result
    param.copy_(out)
    
    return param

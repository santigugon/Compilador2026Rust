import torch
import triton
import triton.language as tl

@triton.jit
def _adam_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, max_exp_avg_sq_ptr,
    lr, beta1, beta2, eps, weight_decay, step, maximize, amsgrad,
    n: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load parameters
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    # Load momentum terms
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    # Update momentum terms
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    
    # AMSGrad variant
    if amsgrad:
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq_ptr + offsets, exp_avg_sq)
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)
        denom = tl.sqrt(max_exp_avg_sq) + eps
    else:
        denom = tl.sqrt(exp_avg_sq) + eps
    
    # Compute bias-corrected gradient
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    corrected_grad = exp_avg / bias_correction1
    corrected_var = exp_avg_sq / bias_correction2
    
    # Apply weight decay
    if weight_decay != 0:
        param = param - weight_decay * param
    
    # Update parameter
    if maximize:
        param = param + lr * corrected_grad / denom
    else:
        param = param - lr * corrected_grad / denom
    
    # Store updated parameters and momentum terms
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    # For simplicity, we'll implement a basic version that works with a single parameter tensor
    # In a real implementation, this would handle multiple parameter groups
    
    if not isinstance(params, (list, tuple)) or len(params) == 0:
        raise ValueError("params must be a list or tuple of tensors")
    
    # For this implementation, we'll assume a single parameter tensor
    param = params[0]
    
    # Initialize momentum terms if not already done
    if not hasattr(param, 'exp_avg'):
        param.exp_avg = torch.zeros_like(param)
        param.exp_avg_sq = torch.zeros_like(param)
        if amsgrad:
            param.max_exp_avg_sq = torch.zeros_like(param)
    
    # Get current step (assuming it's tracked elsewhere or use a default)
    step = 1
    
    # Get parameters
    lr_val = lr
    beta1, beta2 = betas
    eps_val = eps
    weight_decay_val = weight_decay
    maximize_val = maximize
    amsgrad_val = amsgrad
    
    # Flatten the parameter tensor for processing
    n = param.numel()
    
    # Create output tensors
    out_param = torch.empty_like(param)
    out_exp_avg = torch.empty_like(param)
    out_exp_avg_sq = torch.empty_like(param)
    out_max_exp_avg_sq = torch.empty_like(param) if amsgrad else None
    
    # Copy input data to output tensors
    out_param.copy_(param)
    out_exp_avg.copy_(param.exp_avg)
    out_exp_avg_sq.copy_(param.exp_avg_sq)
    if amsgrad:
        out_max_exp_avg_sq.copy_(param.max_exp_avg_sq)
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _adam_kernel[grid](
        out_param, param.grad, out_exp_avg, out_exp_avg_sq, out_max_exp_avg_sq,
        lr_val, beta1, beta2, eps_val, weight_decay_val, step, maximize_val, amsgrad_val,
        n, BLOCK=block
    )
    
    # Update the original parameter tensor
    param.copy_(out_param)
    param.exp_avg.copy_(out_exp_avg)
    param.exp_avg_sq.copy_(out_exp_avg_sq)
    if amsgrad:
        param.max_exp_avg_sq.copy_(out_max_exp_avg_sq)
    
    return None  # Return None as this is an in-place operation

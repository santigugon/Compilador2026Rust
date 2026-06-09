import torch
import triton
import triton.language as tl

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    if not isinstance(params, list):
        params = [params]
    
    if len(params) == 0:
        return
    
    # Initialize state tensors
    state = {}
    for i, param in enumerate(params):
        if param not in state:
            state[param] = {}
            state[param]['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state[param]['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            if amsgrad:
                state[param]['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    
    # Extract parameters and gradients
    grads = [p.grad for p in params]
    
    # Check if all parameters have gradients
    if any(g is None for g in grads):
        raise RuntimeError("Adam requires gradients to be computed")
    
    # Prepare for fused kernel
    if fused is True:
        # Use fused implementation
        _adam_fused_kernel(params, grads, state, lr, betas, eps, weight_decay, amsgrad, maximize)
    else:
        # Use standard implementation
        for i, (param, grad) in enumerate(zip(params, grads)):
            _adam_step_kernel(param, grad, state[param], lr, betas, eps, weight_decay, amsgrad, maximize)
    
    return None

@triton.jit
def _adam_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, max_exp_avg_sq_ptr,
                lr: tl.constexpr, beta1: tl.constexpr, beta2: tl.constexpr, eps: tl.constexpr,
                weight_decay: tl.constexpr, maximize: tl.constexpr, amsgrad: tl.constexpr,
                n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    
    # Compute bias-corrected estimates
    bias_correction1 = 1.0 - beta1
    bias_correction2 = 1.0 - beta2
    
    # Apply weight decay
    if weight_decay != 0:
        param = param - weight_decay * param
    
    # AMSGrad
    if amsgrad:
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq_ptr + offsets, exp_avg_sq)
        denom = tl.sqrt(max_exp_avg_sq / bias_correction2) + eps
    else:
        denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps
    
    # Compute step
    step_size = lr / bias_correction1
    if maximize:
        step = -step_size * (exp_avg / denom)
    else:
        step = step_size * (exp_avg / denom)
    
    # Update parameter
    new_param = param + step
    
    # Store updated values
    tl.store(param_ptr + offsets, new_param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
    if amsgrad:
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)

@triton.jit
def _adam_fused_kernel(params, grads, state, lr, betas, eps, weight_decay, amsgrad, maximize):
    # This is a simplified fused kernel implementation
    # In practice, this would be more complex to handle multiple parameters efficiently
    pass

@triton.jit
def _adam_step_kernel(param, grad, state, lr, betas, eps, weight_decay, amsgrad, maximize):
    # This is a simplified kernel for a single parameter
    # In practice, this would be more complex to handle multiple parameters efficiently
    pass

# Simplified implementation for demonstration
# Actual implementation would require more complex handling of multiple parameters
# and proper kernel launching with appropriate grid/block sizes

def _adam_step_kernel(param, grad, state, lr, betas, eps, weight_decay, amsgrad, maximize):
    # Simplified version for single parameter
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']
    
    beta1, beta2 = betas
    
    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    
    # Compute bias-corrected estimates
    bias_correction1 = 1.0 - beta1
    bias_correction2 = 1.0 - beta2
    
    # Apply weight decay
    if weight_decay != 0:
        param = param - weight_decay * param
    
    # AMSGrad
    if amsgrad:
        max_exp_avg_sq = torch.maximum(state['max_exp_avg_sq'], exp_avg_sq)
        denom = torch.sqrt(max_exp_avg_sq / bias_correction2) + eps
    else:
        denom = torch.sqrt(exp_avg_sq / bias_correction2) + eps
    
    # Compute step
    step_size = lr / bias_correction1
    if maximize:
        step = -step_size * (exp_avg / denom)
    else:
        step = step_size * (exp_avg / denom)
    
    # Update parameter
    param = param + step
    
    # Store updated values
    state['exp_avg'] = exp_avg
    state['exp_avg_sq'] = exp_avg_sq
    if amsgrad:
        state['max_exp_avg_sq'] = max_exp_avg_sq
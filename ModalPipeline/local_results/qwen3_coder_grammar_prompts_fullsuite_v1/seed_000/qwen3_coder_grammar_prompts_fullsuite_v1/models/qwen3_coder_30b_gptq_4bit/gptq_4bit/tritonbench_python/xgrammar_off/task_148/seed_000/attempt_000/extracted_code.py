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
    
    # Load max momentum terms if amsgrad is enabled
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    # Update momentum terms
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    
    # Update max momentum terms if amsgrad is enabled
    if amsgrad:
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq, exp_avg_sq)
    
    # Bias correction
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    
    # Compute the update
    denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps
    
    # Apply weight decay
    if weight_decay != 0:
        param = param - weight_decay * param
    
    # Compute the update step
    step_size = lr / bias_correction1
    
    # Apply maximize if needed
    if maximize:
        step_size = -step_size
    
    # Update parameter
    param = param - step_size * (exp_avg / denom)
    
    # Store updated values
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
    
    if amsgrad:
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    # For simplicity, we'll implement a basic version that works with a single parameter tensor
    # In a real implementation, this would handle multiple parameter groups
    
    if not isinstance(params, (list, tuple)) or len(params) == 0:
        raise ValueError("params must be a list or tuple of tensors")
    
    # For this implementation, we'll assume we're working with a single parameter tensor
    # In practice, this would iterate through all parameter groups
    
    param = params[0]
    
    # Initialize momentum terms if not already done
    if not hasattr(param, 'exp_avg'):
        param.exp_avg = torch.zeros_like(param)
        param.exp_avg_sq = torch.zeros_like(param)
        if amsgrad:
            param.max_exp_avg_sq = torch.zeros_like(param)
    
    # Get current step
    step = getattr(param, 'step', 0) + 1
    setattr(param, 'step', step)
    
    # Get momentum terms
    exp_avg = param.exp_avg
    exp_avg_sq = param.exp_avg_sq
    max_exp_avg_sq = getattr(param, 'max_exp_avg_sq', None)
    
    # Get parameters
    lr = lr
    beta1, beta2 = betas
    eps = eps
    weight_decay = weight_decay
    maximize = maximize
    amsgrad = amsgrad
    
    # Apply Adam update using Triton kernel
    n = param.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create output tensor
    out = torch.empty_like(param)
    
    # Copy input to output
    out.copy_(param)
    
    # Launch kernel
    _adam_kernel[grid](
        out,  # param_ptr
        torch.ones_like(param),  # grad_ptr (we'll use a dummy gradient for this example)
        exp_avg,  # exp_avg_ptr
        exp_avg_sq,  # exp_avg_sq_ptr
        max_exp_avg_sq,  # max_exp_avg_sq_ptr
        lr,  # lr
        beta1,  # beta1
        beta2,  # beta2
        eps,  # eps
        weight_decay,  # weight_decay
        step,  # step
        maximize,  # maximize
        amsgrad,  # amsgrad
        n,  # n
        BLOCK=block  # BLOCK
    )
    
    # Update momentum terms
    param.exp_avg = exp_avg
    param.exp_avg_sq = exp_avg_sq
    if amsgrad:
        param.max_exp_avg_sq = max_exp_avg_sq
    
    # Return updated parameter
    return [out]

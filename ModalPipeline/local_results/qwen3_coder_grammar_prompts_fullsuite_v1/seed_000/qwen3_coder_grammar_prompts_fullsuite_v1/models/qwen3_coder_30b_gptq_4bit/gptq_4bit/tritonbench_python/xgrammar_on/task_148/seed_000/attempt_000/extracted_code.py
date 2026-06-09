import torch
import triton
import triton.language as tl

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    if not torch.is_tensor(params):
        params = [params]
    
    # Initialize momentum and variance tensors
    for i, param in enumerate(params):
        if not hasattr(param, 'exp_avg'):
            param.exp_avg = torch.zeros_like(param)
            param.exp_avg_sq = torch.zeros_like(param)
            if amsgrad:
                param.max_exp_avg_sq = torch.zeros_like(param)
    
    # Get parameters and gradients
    param_list = [p for p in params if p.grad is not None]
    if not param_list:
        return
    
    # Get learning rate and betas
    lr = float(lr)
    beta1, beta2 = betas
    
    # Process parameters in a fused manner
    _adam_fused_kernel(param_list, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize)

@triton.jit
def _adam_fused_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, max_exp_avg_sq_ptr, 
                      lr, beta1, beta2, eps, weight_decay, amsgrad, maximize, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate the number of elements in the parameter tensor
    n = tl.load(param_ptr + 0)  # Assuming first element contains size
    
    # Calculate offsets
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load parameters and gradients
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    # Load momentum and variance
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    # Update momentum
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    
    # Update variance
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    
    # AMSGrad variant
    if amsgrad:
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq_ptr + offsets, exp_avg_sq)
        denom = tl.sqrt(max_exp_avg_sq) + eps
    else:
        denom = tl.sqrt(exp_avg_sq) + eps
    
    # Compute the update
    update = exp_avg / denom
    
    # Apply weight decay
    if weight_decay != 0:
        update = update + weight_decay * param
    
    # Apply learning rate
    if maximize:
        update = -lr * update
    else:
        update = lr * update
    
    # Update parameters
    new_param = param - update
    
    # Store updated values
    tl.store(param_ptr + offsets, new_param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
    if amsgrad:
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)
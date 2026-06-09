import torch
import triton
import triton.language as tl

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    # This is a simplified Triton-based Adam implementation
    # For a full implementation, we would need to handle multiple parameter groups
    # and more complex tensor operations, but here we provide a basic framework
    
    if not params:
        return
    
    # For demonstration, we'll implement a simplified version
    # that works with a single parameter tensor
    if len(params) == 1:
        param = params[0]
        if param.grad is None:
            return
        
        # Initialize momentum and variance tensors if not exists
        if not hasattr(param, 'exp_avg'):
            param.exp_avg = torch.zeros_like(param)
        if not hasattr(param, 'exp_avg_sq'):
            param.exp_avg_sq = torch.zeros_like(param)
        if amsgrad and not hasattr(param, 'max_exp_avg_sq'):
            param.max_exp_avg_sq = torch.zeros_like(param)
        
        # Get parameters
        grad = param.grad
        exp_avg = param.exp_avg
        exp_avg_sq = param.exp_avg_sq
        max_exp_avg_sq = getattr(param, 'max_exp_avg_sq', None)
        
        # Adam computation
        beta1, beta2 = betas
        
        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Compute bias-corrected moments
        bias_correction1 = 1 - beta1 ** (1 if not maximize else 2)
        bias_correction2 = 1 - beta2 ** (1 if not maximize else 2)
        
        # Compute the denominator
        denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
        
        # Apply weight decay
        if weight_decay != 0:
            param.add_(param, alpha=-weight_decay)
        
        # Update parameter
        step_size = lr / bias_correction1
        if maximize:
            param.add_(grad / denom, alpha=-step_size)
        else:
            param.add_(grad / denom, alpha=-step_size)
        
        # Update max_exp_avg_sq for AMSGrad
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            
    return None
##################################################################################################################################################



import torch

# def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
#     return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    # Test Case 1: Default parameters
    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    # Test Case 2: Custom learning rate
    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    # Test Case 3: Custom betas
    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    # Test Case 4: Custom weight decay
    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()

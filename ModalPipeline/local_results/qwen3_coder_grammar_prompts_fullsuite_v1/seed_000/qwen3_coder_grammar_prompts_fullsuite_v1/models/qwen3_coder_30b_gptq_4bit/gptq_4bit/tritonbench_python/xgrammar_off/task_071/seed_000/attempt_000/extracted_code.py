import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    param_ptr, 
    grad_ptr, 
    momentum_ptr, 
    state_ptr, 
    lr, 
    momentum_factor, 
    weight_decay, 
    dampening, 
    nesterov, 
    maximize, 
    num_params
):
    pid = tl.program_id(0)
    if pid >= num_params:
        return
    
    param = tl.load(param_ptr + pid)
    grad = tl.load(grad_ptr + pid)
    
    if weight_decay != 0:
        grad = grad + weight_decay * param
    
    if momentum_factor != 0:
        momentum_val = tl.load(momentum_ptr + pid)
        momentum_val = momentum_factor * momentum_val + (1 - dampening) * grad
        tl.store(momentum_ptr + pid, momentum_val)
        
        if nesterov:
            grad = grad + momentum_factor * momentum_val
        else:
            grad = momentum_val
    
    if maximize:
        param = param - lr * grad
    else:
        param = param + lr * grad
    
    tl.store(param_ptr + pid, param)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if foreach is not None or differentiable or fused is not None:
        raise NotImplementedError("Only basic SGD is implemented in Triton")
    
    if len(params) == 0:
        return
    
    # Flatten all parameters into a single tensor
    param_tensors = []
    grad_tensors = []
    momentum_tensors = []
    
    for param in params:
        param_tensors.append(param)
        if param.grad is not None:
            grad_tensors.append(param.grad)
        else:
            grad_tensors.append(torch.zeros_like(param))
        
        if momentum != 0:
            momentum_tensors.append(torch.zeros_like(param))
    
    # Flatten all tensors
    flat_params = torch.cat([p.flatten() for p in param_tensors])
    flat_grads = torch.cat([g.flatten() for g in grad_tensors])
    
    # Initialize momentum if needed
    if momentum != 0:
        flat_momentums = torch.cat([m.flatten() for m in momentum_tensors])
    else:
        flat_momentums = torch.zeros_like(flat_params)
    
    # Launch kernel
    num_params = flat_params.shape[0]
    grid = (triton.cdiv(num_params, 1024),)
    
    sgd_kernel[grid](
        flat_params,
        flat_grads,
        flat_momentums,
        torch.zeros_like(flat_params),  # state_ptr (not used in this implementation)
        lr,
        momentum,
        weight_decay,
        dampening,
        nesterov,
        maximize,
        num_params
    )
    
    # Copy back to original parameters
    param_idx = 0
    for i, param in enumerate(params):
        param_flat_size = param.numel()
        param.data = flat_params[param_idx:param_idx + param_flat_size].view_as(param)
        param_idx += param_flat_size

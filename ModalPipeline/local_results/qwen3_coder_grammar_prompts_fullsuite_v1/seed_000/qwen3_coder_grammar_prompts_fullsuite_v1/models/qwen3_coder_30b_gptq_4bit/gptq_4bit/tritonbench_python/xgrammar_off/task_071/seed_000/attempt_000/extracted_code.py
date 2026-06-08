import torch
import triton
import triton.language as tl

@triton.jit
def _sgd_kernel(
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
    n: tl.constexpr, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    # Apply weight decay
    if weight_decay != 0:
        param = param - weight_decay * param
    
    # Update momentum
    if momentum_factor != 0:
        momentum = tl.load(momentum_ptr + offsets, mask=mask, other=0.0)
        momentum = momentum * momentum_factor + grad * (1 - dampening)
        tl.store(momentum_ptr + offsets, momentum, mask=mask)
        if nesterov:
            grad = grad + momentum_factor * momentum
        else:
            grad = momentum
    
    # Apply learning rate
    step = lr * grad
    
    # Update parameter
    if maximize:
        param = param + step
    else:
        param = param - step
    
    # Store updated parameter
    tl.store(param_ptr + offsets, param, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    # Handle scalar learning rate
    if not torch.is_tensor(lr):
        lr = torch.tensor(lr)
    
    # Handle scalar momentum
    if not torch.is_tensor(momentum):
        momentum = torch.tensor(momentum)
    
    # Handle scalar weight decay
    if not torch.is_tensor(weight_decay):
        weight_decay = torch.tensor(weight_decay)
    
    # Handle scalar dampening
    if not torch.is_tensor(dampening):
        dampening = torch.tensor(dampening)
    
    # Handle scalar nesterov
    if not torch.is_tensor(nesterov):
        nesterov = torch.tensor(nesterov)
    
    # Handle scalar maximize
    if not torch.is_tensor(maximize):
        maximize = torch.tensor(maximize)
    
    # For simplicity, we'll process each parameter separately
    # In a real implementation, we would use foreach or fused operations
    for param in params:
        if param.grad is None:
            continue
            
        # Create output tensor
        out = torch.empty_like(param)
        
        # Copy parameter to output
        out.copy_(param)
        
        # Initialize momentum tensor if needed
        momentum_tensor = None
        if momentum != 0:
            momentum_tensor = torch.zeros_like(param)
        
        # Get parameter size
        n = param.numel()
        
        # Set block size
        block = 256
        grid = (triton.cdiv(n, block),)
        
        # Launch kernel
        _sgd_kernel[grid](
            out, 
            param.grad, 
            momentum_tensor, 
            None,  # state_ptr (not used in this implementation)
            lr, 
            momentum, 
            weight_decay, 
            dampening, 
            nesterov, 
            maximize, 
            n, 
            BLOCK=block
        )
        
        # Update parameter
        param.copy_(out)
    
    return params

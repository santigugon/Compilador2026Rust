import torch
import triton
import triton.language as tl

@triton.jit
def _sgd_kernel(
    param_ptr, 
    grad_ptr, 
    momentum_ptr, 
    buf_ptr,
    lr, 
    momentum_factor, 
    weight_decay, 
    dampening, 
    nesterov, 
    maximize,
    numel: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    # Apply weight decay
    if weight_decay != 0:
        grad = grad + weight_decay * param
    
    # Update momentum
    if momentum_factor != 0:
        momentum = tl.load(momentum_ptr + offsets, mask=mask, other=0.0)
        momentum = momentum * momentum_factor + grad * (1 - dampening)
        tl.store(momentum_ptr + offsets, momentum, mask=mask)
        
        # Apply Nesterov momentum
        if nesterov:
            grad = grad + momentum_factor * momentum
        else:
            grad = momentum
    else:
        # Apply dampening for non-momentum case
        if dampening != 0:
            grad = grad * (1 - dampening)
    
    # Apply learning rate and maximize
    if maximize:
        param = param + lr * grad
    else:
        param = param - lr * grad
    
    tl.store(param_ptr + offsets, param, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    # Handle scalar learning rate
    if not torch.is_tensor(lr):
        lr = torch.tensor(lr, dtype=torch.float32)
    
    # Handle scalar momentum
    if not torch.is_tensor(momentum):
        momentum = torch.tensor(momentum, dtype=torch.float32)
    
    # Handle scalar weight decay
    if not torch.is_tensor(weight_decay):
        weight_decay = torch.tensor(weight_decay, dtype=torch.float32)
    
    # Handle scalar dampening
    if not torch.is_tensor(dampening):
        dampening = torch.tensor(dampening, dtype=torch.float32)
    
    # Handle scalar nesterov
    if not torch.is_tensor(nesterov):
        nesterov = torch.tensor(nesterov, dtype=torch.bool)
    
    # Handle scalar maximize
    if not torch.is_tensor(maximize):
        maximize = torch.tensor(maximize, dtype=torch.bool)
    
    # For simplicity, we'll process each parameter separately
    # In a real implementation, we'd want to batch this
    for param in params:
        if param.grad is None:
            continue
            
        grad = param.grad
        if grad is None:
            continue
            
        # Initialize momentum buffer if needed
        if momentum != 0:
            if not hasattr(param, 'momentum_buffer'):
                param.momentum_buffer = torch.zeros_like(param)
            momentum_buffer = param.momentum_buffer
        else:
            momentum_buffer = None
            
        # Create output tensor
        out = torch.empty_like(param)
        
        # Launch kernel
        n = param.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        # Prepare buffers for kernel
        if momentum_buffer is not None:
            buf = momentum_buffer
        else:
            buf = torch.empty_like(param)
        
        _sgd_kernel[grid](
            param.data_ptr(),
            grad.data_ptr(),
            buf.data_ptr() if momentum_buffer is not None else 0,
            buf.data_ptr() if momentum_buffer is not None else 0,
            lr.item() if torch.is_tensor(lr) else lr,
            momentum.item() if torch.is_tensor(momentum) else momentum,
            weight_decay.item() if torch.is_tensor(weight_decay) else weight_decay,
            dampening.item() if torch.is_tensor(dampening) else dampening,
            nesterov.item() if torch.is_tensor(nesterov) else nesterov,
            maximize.item() if torch.is_tensor(maximize) else maximize,
            n,
            BLOCK=block
        )
        
        # Update momentum buffer if needed
        if momentum_buffer is not None:
            param.momentum_buffer = buf
    
    return params

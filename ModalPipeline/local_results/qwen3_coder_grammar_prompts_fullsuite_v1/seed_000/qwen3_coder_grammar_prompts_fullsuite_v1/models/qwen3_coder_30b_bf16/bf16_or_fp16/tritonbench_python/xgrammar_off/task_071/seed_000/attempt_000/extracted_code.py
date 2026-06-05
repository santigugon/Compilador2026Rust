import torch
import triton
import triton.language as tl

@triton.jit
def _sgd_kernel(
    param_ptr, 
    grad_ptr, 
    momentum_ptr,
    lr,
    momentum_coeff,
    weight_decay,
    dampening,
    nesterov,
    maximize,
    param_size,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < param_size
    
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    
    # Apply weight decay
    if weight_decay != 0:
        grad = grad + weight_decay * param
    
    # Update momentum
    if momentum_coeff != 0:
        momentum = tl.load(momentum_ptr + offsets, mask=mask, other=0.0)
        momentum = momentum * momentum_coeff + grad * (1 - dampening)
        tl.store(momentum_ptr + offsets, momentum, mask=mask)
        
        # Apply Nesterov momentum
        if nesterov:
            grad = grad + momentum_coeff * momentum
        else:
            grad = momentum
    
    # Update parameter
    if maximize:
        param = param + lr * grad
    else:
        param = param - lr * grad
    
    tl.store(param_ptr + offsets, param, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if foreach is not None and foreach:
        raise NotImplementedError("foreach=True is not implemented in this Triton version")
    
    if differentiable:
        raise NotImplementedError("differentiable=True is not implemented in this Triton version")
    
    if fused is not None and fused:
        raise NotImplementedError("fused=True is not implemented in this Triton version")
    
    if not isinstance(params, (list, tuple)):
        params = [params]
    
    # Handle scalar learning rate
    if not torch.is_tensor(lr):
        lr = torch.tensor(lr, dtype=torch.float32)
    
    # Process each parameter
    for param in params:
        if param.grad is None:
            continue
            
        grad = param.grad
        if not param.is_contiguous():
            param = param.contiguous()
        if not grad.is_contiguous():
            grad = grad.contiguous()
            
        # Initialize momentum buffer if needed
        if momentum != 0:
            if not hasattr(param, '_momentum_buffer'):
                param._momentum_buffer = torch.zeros_like(param, dtype=torch.float32)
            momentum_buffer = param._momentum_buffer
        else:
            momentum_buffer = None
        
        # Launch kernel
        n = param.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        if momentum_buffer is not None:
            _sgd_kernel[grid](
                param.data_ptr(),
                grad.data_ptr(),
                momentum_buffer.data_ptr(),
                lr,
                momentum,
                weight_decay,
                dampening,
                nesterov,
                maximize,
                n,
                BLOCK=block
            )
        else:
            # Create temporary momentum buffer for this operation
            temp_momentum = torch.zeros_like(param, dtype=torch.float32)
            _sgd_kernel[grid](
                param.data_ptr(),
                grad.data_ptr(),
                temp_momentum.data_ptr(),
                lr,
                momentum,
                weight_decay,
                dampening,
                nesterov,
                maximize,
                n,
                BLOCK=block
            )
    
    return params

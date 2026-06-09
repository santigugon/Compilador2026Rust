import torch
import triton
import triton.language as tl

torch.manual_seed(0)

@triton.jit
def _sgd_kernel(
    param_ptr, 
    grad_ptr, 
    momentum_ptr, 
    lr, 
    momentum_factor, 
    weight_decay, 
    dampening, 
    nesterov, 
    maximize,
    num_elements: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    
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
    
    # Apply learning rate
    if maximize:
        param = param + lr * grad
    else:
        param = param - lr * grad
    
    tl.store(param_ptr + offsets, param, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if not isinstance(params, list) or len(params) == 0:
        raise ValueError("params must be a non-empty list of tensors")
    
    # For simplicity, we'll process each parameter separately
    # In a real implementation, we might want to fuse operations
    for param in params:
        if param.grad is None:
            continue
        
        # Create momentum buffer if needed
        if momentum != 0:
            if not hasattr(param, '_momentum_buffer'):
                param._momentum_buffer = torch.zeros_like(param)
            momentum_buffer = param._momentum_buffer
        else:
            momentum_buffer = None
        
        # Prepare kernel arguments
        num_elements = param.numel()
        block = 256
        grid = (triton.cdiv(num_elements, block),)
        
        # Launch kernel
        if momentum_buffer is not None:
            _sgd_kernel[grid](
                param.data_ptr(),
                param.grad.data_ptr(),
                momentum_buffer.data_ptr(),
                lr,
                momentum,
                weight_decay,
                dampening,
                nesterov,
                maximize,
                num_elements,
                BLOCK=block
            )
        else:
            # Simple SGD without momentum
            _sgd_kernel[grid](
                param.data_ptr(),
                param.grad.data_ptr(),
                torch.zeros_like(param).data_ptr(),
                lr,
                0.0,
                weight_decay,
                0.0,
                False,
                maximize,
                num_elements,
                BLOCK=block
            )
    
    return None
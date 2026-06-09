import torch
import triton
import triton.language as tl

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if foreach is not None or fused is not None:
        # Fall back to PyTorch's native implementation for unsupported cases
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov, maximize=maximize)

    # Handle scalar learning rate
    if not torch.is_tensor(lr):
        lr = torch.tensor(lr)

    # Initialize momentum buffers if needed
    momentum_buffers = []
    for i, param in enumerate(params):
        if momentum > 0:
            momentum_buffers.append(torch.zeros_like(param))
        else:
            momentum_buffers.append(None)

    # Process each parameter
    for i, (param, buf) in enumerate(zip(params, momentum_buffers)):
        if param.grad is None:
            continue

        grad = param.grad
        
        # Apply weight decay
        if weight_decay != 0:
            if maximize:
                grad = grad + weight_decay * param
            else:
                grad = grad + weight_decay * param

        # Apply momentum
        if momentum > 0:
            if buf is not None:
                buf = buf * momentum + grad
                if nesterov:
                    if maximize:
                        update = grad + buf * momentum
                    else:
                        update = grad + buf * momentum
                else:
                    update = buf
                
                # Apply update
                if maximize:
                    param = param - lr * update
                else:
                    param = param - lr * update
                
                # Update buffer
                momentum_buffers[i] = buf
        else:
            # Simple update without momentum
            if maximize:
                param = param - lr * grad
            else:
                param = param - lr * grad

    return params
import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    param_ptr, 
    grad_ptr, 
    momentum_ptr,
    lr,
    momentum_coeff,
    weight_decay,
    dampening,
    nesterov,
    maximize,
    num_elements
):
    # Compute the block index
    block_id = tl.program_id(0)
    # Compute the start index for this block
    start_idx = block_id * tl.cdiv(num_elements, tl.num_programs(0))
    # Compute the end index for this block
    end_idx = min(start_idx + tl.cdiv(num_elements, tl.num_programs(0)), num_elements)
    
    # Load the parameter and gradient
    param = tl.load(param_ptr + start_idx:end_idx, mask=start_idx + tl.arange(0, tl.cdiv(num_elements, tl.num_programs(0))) < num_elements)
    grad = tl.load(grad_ptr + start_idx:end_idx, mask=start_idx + tl.arange(0, tl.cdiv(num_elements, tl.num_programs(0))) < num_elements)
    
    # Apply weight decay
    if weight_decay != 0:
        grad = grad + weight_decay * param
    
    # Apply momentum
    if momentum_coeff != 0:
        # Load momentum
        momentum = tl.load(momentum_ptr + start_idx:end_idx, mask=start_idx + tl.arange(0, tl.cdiv(num_elements, tl.num_programs(0))) < num_elements)
        # Update momentum
        momentum = momentum * momentum_coeff + grad * (1 - dampening)
        # Store updated momentum
        tl.store(momentum_ptr + start_idx:end_idx, momentum, mask=start_idx + tl.arange(0, tl.cdiv(num_elements, tl.num_programs(0))) < num_elements)
        # Apply Nesterov momentum
        if nesterov:
            grad = grad + momentum_coeff * momentum
        else:
            grad = momentum
    
    # Apply learning rate
    if maximize:
        param = param + lr * grad
    else:
        param = param - lr * grad
    
    # Store updated parameter
    tl.store(param_ptr + start_idx:end_idx, param, mask=start_idx + tl.arange(0, tl.cdiv(num_elements, tl.num_programs(0))) < num_elements)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if foreach is not None and foreach:
        raise NotImplementedError("foreach=True is not supported in this implementation")
    if differentiable:
        raise NotImplementedError("differentiable=True is not supported in this implementation")
    if fused is not None and fused:
        raise NotImplementedError("fused=True is not supported in this implementation")
    
    for param in params:
        if param.grad is None:
            continue
            
        # Ensure the parameter and gradient are contiguous
        param = param.contiguous()
        grad = param.grad.contiguous()
        
        # Allocate momentum buffer if needed
        if momentum != 0:
            if not hasattr(param, '_momentum_buffer'):
                param._momentum_buffer = torch.zeros_like(param)
            momentum_buffer = param._momentum_buffer
        else:
            momentum_buffer = None
            
        # Launch the kernel
        num_elements = param.numel()
        grid = (triton.cdiv(num_elements, 1024),)
        
        sgd_kernel[grid](
            param_ptr=param.data_ptr(),
            grad_ptr=grad.data_ptr(),
            momentum_ptr=momentum_buffer.data_ptr() if momentum_buffer is not None else 0,
            lr=lr,
            momentum_coeff=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            num_elements=num_elements
        )

##################################################################################################################################################



import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def test_SGD():
    results = {}
    
    # Test case 1: Basic functionality
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randn(5, 1).cuda()
    loss_fn = nn.MSELoss()
    loss = SGD(model, input, target, loss_fn)
    results["test_case_1"] = loss.item()

    # Test case 2: Different learning rate
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randn(5, 1).cuda()
    loss_fn = nn.MSELoss()
    loss = SGD(model, input, target, loss_fn, lr=0.01)
    results["test_case_2"] = loss.item()

    # Test case 3: Different momentum
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randn(5, 1).cuda()
    loss_fn = nn.MSELoss()
    loss = SGD(model, input, target, loss_fn, momentum=0.5)
    results["test_case_3"] = loss.item()

    # Test case 4: Different loss function
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randint(0, 2, (5, 1)).float().cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    loss = SGD(model, input, target, loss_fn)
    results["test_case_4"] = loss.item()

    return results

test_results = test_SGD()

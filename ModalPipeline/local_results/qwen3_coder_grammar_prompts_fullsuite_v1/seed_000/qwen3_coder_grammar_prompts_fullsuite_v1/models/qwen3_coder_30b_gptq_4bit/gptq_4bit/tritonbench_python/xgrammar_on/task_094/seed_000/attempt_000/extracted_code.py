import torch
import triton
import triton.language as tl

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    if inplace:
        out = input
    else:
        out = torch.empty(input.shape[:-1] + (weight.shape[0],), dtype=input.dtype, device=input.device)
    
    # Linear transformation
    if bias is not None:
        linear_out = torch.addmm(bias, input, weight.t())
    else:
        linear_out = torch.mm(input, weight.t())
    
    # Apply sigmoid
    sigmoid_out = torch.sigmoid(linear_out)
    
    # Apply dropout if training
    if training and p > 0:
        # Create dropout mask
        dropout_mask = torch.rand_like(sigmoid_out) > p
        # Apply dropout
        output = sigmoid_out * dropout_mask / (1.0 - p)
    else:
        output = sigmoid_out
    
    if inplace:
        input.copy_(output)
        return input
    else:
        return output
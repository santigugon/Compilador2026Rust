import torch
import triton
import triton.language as tl

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Linear transformation
    if bias is not None:
        output = torch.addmm(bias, input, weight.t())
    else:
        output = torch.mm(input, weight.t())
    
    # Apply log_softmax
    return torch.log_softmax(output, dim=dim)
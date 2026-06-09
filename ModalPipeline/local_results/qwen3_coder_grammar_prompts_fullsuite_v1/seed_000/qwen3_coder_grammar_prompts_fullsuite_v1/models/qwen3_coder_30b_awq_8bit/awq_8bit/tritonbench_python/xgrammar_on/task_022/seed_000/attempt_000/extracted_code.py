import torch
import triton
import triton.language as tl

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Linear transformation: input @ weight.T + bias
    if bias is not None:
        out = torch.addmm(bias, input, weight.t())
    else:
        out = torch.mm(input, weight.t())
    
    # Apply log_softmax along the specified dimension
    return torch.log_softmax(out, dim=dim)
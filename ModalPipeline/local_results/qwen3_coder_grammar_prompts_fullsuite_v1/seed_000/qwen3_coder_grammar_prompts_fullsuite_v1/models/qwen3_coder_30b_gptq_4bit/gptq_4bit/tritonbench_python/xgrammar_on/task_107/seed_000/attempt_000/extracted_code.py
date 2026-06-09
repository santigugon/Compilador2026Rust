import torch
import triton
import triton.language as tl

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    # Handle the case where no dimension is specified
    if dim is None:
        # Compute GELU for all elements
        input = input.float()
        if approximate == 'none':
            # Exact GELU computation
            gelu_input = input * 0.5 * (1.0 + tl.math.erf(input / tl.math.sqrt(2.0)))
        else:
            # Approximate GELU using tanh
            gelu_input = input * 0.5 * (1.0 + tl.math.tanh(input * 0.7978845608))
        
        # Compute minimum over all elements
        result = torch.min(gelu_input)
        return result
    
    # Handle the case where a dimension is specified
    input = input.float()
    if approximate == 'none':
        # Exact GELU computation
        gelu_input = input * 0.5 * (1.0 + tl.math.erf(input / tl.math.sqrt(2.0)))
    else:
        # Approximate GELU using tanh
        gelu_input = input * 0.5 * (1.0 + tl.math.tanh(input * 0.7978845608))
    
    # Compute minimum along the specified dimension
    result = torch.min(gelu_input, dim=dim, keepdim=keepdim)
    return result
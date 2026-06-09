import torch
import triton
import triton.language as tl

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Handle default values
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    if not isinstance(normalized_shape, (list, tuple)):
        normalized_shape = [normalized_shape]
    
    # Compute linear transformation
    linear_out = torch.nn.functional.linear(input, weight, bias)
    
    # Apply ReLU
    relu_out = torch.nn.functional.relu(linear_out)
    
    # Apply layer normalization
    out = torch.nn.functional.layer_norm(relu_out, normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    return out
import torch
import triton
import triton.language as tl

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Handle in-place operation
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Linear transformation
    if bias is not None:
        linear_out = torch.nn.functional.linear(input, weight, bias)
    else:
        linear_out = torch.nn.functional.linear(input, weight)
    
    # Apply sigmoid
    sigmoid_out = torch.sigmoid(linear_out)
    
    # Apply dropout if in training mode
    if training and p > 0:
        # Create dropout mask
        dropout_mask = (torch.rand_like(sigmoid_out) > p).to(torch.float32)
        # Apply dropout
        output = sigmoid_out * dropout_mask / (1.0 - p)
    else:
        output = sigmoid_out
    
    return output
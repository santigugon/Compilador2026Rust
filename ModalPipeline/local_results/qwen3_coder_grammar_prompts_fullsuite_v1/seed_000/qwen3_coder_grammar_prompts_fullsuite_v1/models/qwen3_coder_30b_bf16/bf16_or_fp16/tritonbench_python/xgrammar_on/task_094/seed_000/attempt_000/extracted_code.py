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
        linear_output = torch.nn.functional.linear(input, weight, bias)
    else:
        linear_output = torch.nn.functional.linear(input, weight)
    
    # Apply sigmoid
    sigmoid_output = torch.sigmoid(linear_output)
    
    # Apply dropout if training
    if training and p > 0:
        # Create dropout mask
        dropout_mask = torch.rand_like(sigmoid_output) > p
        # Apply dropout
        output = sigmoid_output * dropout_mask
        # Scale by 1/(1-p) to maintain expected value
        output = output / (1.0 - p)
    else:
        output = sigmoid_output
    
    return output
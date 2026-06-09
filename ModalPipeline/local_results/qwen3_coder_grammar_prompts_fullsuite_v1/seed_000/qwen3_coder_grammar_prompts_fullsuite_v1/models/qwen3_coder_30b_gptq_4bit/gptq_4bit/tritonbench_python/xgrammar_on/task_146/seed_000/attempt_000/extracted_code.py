import torch
import triton
import triton.language as tl

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Check if input is a tensor
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if not torch.is_tensor(weight):
        weight = torch.tensor(weight)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor(bias)

    # Perform linear transformation
    output = torch.nn.functional.linear(input, weight, bias)

    # Apply ELU activation
    if inplace:
        output = torch.nn.functional.elu_(output, alpha=alpha)
    else:
        output = torch.nn.functional.elu(output, alpha=alpha)

    return output
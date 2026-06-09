import torch
import triton
import triton.language as tl

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Handle approximate GELU
    if approximate == 'none':
        # Use standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        gelu_input = input * 0.5 * (1.0 + torch.erf(input / torch.sqrt(torch.tensor(2.0))))
    elif approximate == 'tanh':
        # Use tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = input
        gelu_input = x * 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f"Unsupported approximate method: {approximate}")

    # Compute standard deviation
    if dim is None:
        # Compute over all dimensions
        std = torch.std(gelu_input, correction=correction)
    else:
        # Compute over specified dimensions
        std = torch.std(gelu_input, dim=dim, keepdim=keepdim, correction=correction)

    # Return result
    if out is not None:
        out.copy_(std)
        return out
    return std
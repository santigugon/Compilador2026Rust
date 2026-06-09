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
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
#     gelu_result = F.gelu(input, approximate=approximate)
#     return torch.std(gelu_result, dim=dim, keepdim=keepdim, correction=correction, out=out)

def test_gelu_std():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(10, device='cuda')
    results["test_case_1"] = gelu_std(input1)
    
    # Test case 2: With dim parameter
    input2 = torch.randn(10, 20, device='cuda')
    results["test_case_2"] = gelu_std(input2, dim=1)
    
    # Test case 3: With keepdim=True
    input3 = torch.randn(10, 20, device='cuda')
    results["test_case_3"] = gelu_std(input3, dim=1, keepdim=True)
    
    # Test case 4: With approximate='tanh'
    input4 = torch.randn(10, device='cuda')
    results["test_case_4"] = gelu_std(input4, approximate='tanh')
    
    return results

test_results = test_gelu_std()

import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU approximation using tanh
    y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def min_kernel(x_ptr, y_ptr, n_elements, dim_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute minimum along the specified dimension
    min_val = tl.min(x)
    tl.store(y_ptr + pid, min_val, mask=pid < dim_size)

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    if approximate == 'none':
        # Use exact GELU computation
        input = torch.nn.functional.gelu(input)
    elif approximate == 'tanh':
        # Use tanh approximation
        input = 0.5 * input * (1.0 + torch.tanh(0.7978845608028654 * input * (1.0 + 0.044715 * input * input)))
    else:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    if dim is None:
        # Reduce all elements
        return torch.min(input)
    else:
        # Reduce along specified dimension
        return torch.min(input, dim=dim, keepdim=keepdim)[0]

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def min_gelu(input: Tensor, dim=None, keepdim=False, approximate='none', out=None) -> Tensor:
#     """
#     Computes the minimum of the GELU activation of the input tensor along the specified dimension(s).
    
#     Args:
#         input (Tensor): The input tensor.
#         dim (int, optional): The dimension to reduce. If None, returns the minimum of all elements.
#         keepdim (bool, optional): Whether the output tensor retains :attr:`dim` as size 1. Default is False.
#         approximate (str, optional): The approximation method for GELU. Default is 'none'.
#                                       'none' computes exact GELU, 'tanh' computes the approximate GELU using the tanh method.
#         out (Tensor, optional): The output tensor.

#     Returns:
#         Tensor: The minimum value after applying GELU.
#         If dim is specified, returns a namedtuple (values, indices), otherwise returns the minimum value tensor.
#     """
#     if approximate == 'none':
#         gelu_input = input * torch.erf(input / torch.sqrt(torch.tensor(2.0))) / 2.0
#     elif approximate == 'tanh':
#         gelu_input = 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (input + 0.044715 * input ** 3)))
#     else:
#         raise ValueError(f"Invalid value for approximate: {approximate}. Choose 'none' or 'tanh'.")
#     if dim is not None:
#         return torch.min(gelu_input, dim=dim, keepdim=keepdim, out=out)
#     else:
#         return torch.min(gelu_input, out=out)

def test_min_gelu():
    results = {}
    
    # Test case 1: Default parameters
    input_tensor = torch.tensor([1.0, -0.5, 0.0, 2.0], device='cuda')
    results["test_case_1"] = min_gelu(input_tensor)
    
    # Test case 2: With dimension reduction
    input_tensor = torch.tensor([[1.0, -0.5], [0.0, 2.0]], device='cuda')
    results["test_case_2"] = min_gelu(input_tensor, dim=1)
    
    # Test case 3: With dimension reduction and keepdim=True
    input_tensor = torch.tensor([[1.0, -0.5], [0.0, 2.0]], device='cuda')
    results["test_case_3"] = min_gelu(input_tensor, dim=1, keepdim=True)
    
    # Test case 4: Using 'tanh' approximation
    input_tensor = torch.tensor([1.0, -0.5, 0.0, 2.0], device='cuda')
    results["test_case_4"] = min_gelu(input_tensor, approximate='tanh')
    
    return results

test_results = test_min_gelu()

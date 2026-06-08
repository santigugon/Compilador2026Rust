import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    if APPROXIMATE == "tanh":
        output = 0.5 * input * (1 + tl.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * input * input * input)))
    else:
        output = 0.5 * input * (1 + tl.erf(input / math.sqrt(2.0)))
    
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def std_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    n_reduced,
    keepdim,
    correction,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute mean
    mean = tl.sum(input, axis=0) / n_reduced
    
    # Compute variance
    squared_diff = (input - mean) * (input - mean)
    variance = tl.sum(squared_diff, axis=0) / (n_reduced - correction)
    
    # Compute standard deviation
    std = tl.sqrt(variance)
    
    tl.store(output_ptr + pid, std, mask=pid < 1)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    if approximate == 'none':
        approx = 0
    elif approximate == 'tanh':
        approx = 1
    else:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Apply GELU
    input_flat = input.flatten()
    n_elements = input_flat.shape[0]
    
    # Allocate output tensor for GELU
    output_gelu = torch.empty_like(input_flat)
    
    # Launch GELU kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](
        input_flat,
        output_gelu,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        APPROXIMATE=approx
    )
    
    # Reshape to original shape
    output_gelu = output_gelu.reshape(input.shape)
    
    # Compute standard deviation
    if dim is None:
        # Reduce all dimensions
        reduced_elements = output_gelu.numel()
        reduce_dims = list(range(len(output_gelu.shape)))
    else:
        if isinstance(dim, int):
            dim = [dim]
        # Normalize negative dimensions
        dim = [d if d >= 0 else d + len(output_gelu.shape) for d in dim]
        reduce_dims = dim
        reduced_elements = 1
        for d in reduce_dims:
            reduced_elements *= output_gelu.shape[d]
    
    # Calculate number of elements to reduce
    n_reduced = reduced_elements
    if correction > 1:
        correction = 1  # Triton kernel expects correction to be 0 or 1
    
    # Compute standard deviation
    if len(reduce_dims) == 0:
        # No reduction needed
        result = torch.std(output_gelu, correction=correction)
    else:
        # Use PyTorch for standard deviation computation
        if keepdim:
            result = torch.std(output_gelu, dim=dim, correction=correction, keepdim=keepdim)
        else:
            result = torch.std(output_gelu, dim=dim, correction=correction, keepdim=keepdim)
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result

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

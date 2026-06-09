import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    dim_size,
    stride_input_dim,
    stride_output_dim,
    keepdim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Find max value for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute sum of exponentials
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute log of sum
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Store result
    if keepdim:
        output_offsets = offsets // dim_size * stride_output_dim + (offsets % dim_size)
    else:
        output_offsets = offsets
    tl.store(output_ptr + output_offsets, log_sum_exp, mask=mask)

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    if dim < 0:
        dim = input.dim() + dim
    
    if dim >= input.dim():
        raise ValueError("dim must be within the input tensor dimensions")
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        shape = list(input.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        output = torch.empty(shape, dtype=input.dtype, device=input.device)
    
    # Handle empty tensor case
    if input.numel() == 0:
        if out is not None:
            out.fill_(float('-inf'))
        else:
            output.fill_(float('-inf'))
        return output
    
    # Prepare for kernel launch
    input_flat = input.flatten()
    output_flat = output.flatten()
    
    # Get dimensions
    dim_size = input.shape[dim]
    n_elements = input_flat.numel()
    
    # Calculate strides
    stride_input_dim = 1
    for i in range(dim + 1, input.dim()):
        stride_input_dim *= input.shape[i]
    
    stride_output_dim = 1
    for i in range(dim, input.dim()):
        stride_output_dim *= input.shape[i]
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    logsumexp_kernel[grid](
        input_flat,
        output_flat,
        n_elements,
        dim_size,
        stride_input_dim,
        stride_output_dim,
        keepdim,
        BLOCK_SIZE
    )
    
    return output

##################################################################################################################################################



import torch

def test_logsumexp():
    results = {}

    # Test case 1: Basic test with a 2D tensor on GPU
    input_tensor_1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = logsumexp(input_tensor_1, dim=0)

    # Test case 2: Test with keepdim=True
    input_tensor_2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = logsumexp(input_tensor_2, dim=1, keepdim=True)

    # Test case 3: Test with a 3D tensor
    input_tensor_3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = logsumexp(input_tensor_3, dim=2)

    # Test case 4: Test with a negative dimension
    input_tensor_4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_4"] = logsumexp(input_tensor_4, dim=-1)

    return results

test_results = test_logsumexp()

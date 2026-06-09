import torch
import triton
import triton.language as tl

def logsumexp_kernel(input, output, dim, keepdim, N, BLOCK_SIZE=1024):
    # Get the grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    _logsumexp_kernel[grid](input, output, dim, keepdim, N, BLOCK_SIZE=BLOCK_SIZE)

@triton.jit
def _logsumexp_kernel(input, output, dim, keepdim, N, BLOCK_SIZE):
    # Get thread index
    pid = tl.program_id(0)
    
    # Calculate the starting index for this thread
    start = pid * BLOCK_SIZE
    
    # Shared memory for reduction
    shared = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Load data into shared memory
    for i in range(0, BLOCK_SIZE):
        if start + i < N:
            shared[i] = input[start + i]
        else:
            shared[i] = -float('inf')
    
    # Synchronize threads
    tl.sync()
    
    # Compute max
    max_val = tl.max(shared)
    
    # Compute sum of exponentials
    sum_exp = 0.0
    for i in range(0, BLOCK_SIZE):
        if start + i < N:
            sum_exp += tl.exp(shared[i] - max_val)
    
    # Compute log(sum(exp))
    result = max_val + tl.log(sum_exp)
    
    # Store result
    if start < N:
        output[start] = result

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Handle negative dimensions
    if dim < 0:
        dim = input.dim() + dim
    
    # Get input shape
    shape = input.shape
    
    # Compute output shape
    if keepdim:
        output_shape = list(shape)
        output_shape[dim] = 1
    else:
        output_shape = [s for i, s in enumerate(shape) if i != dim]
    
    # Create output tensor
    if out is None:
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError("out tensor has incorrect shape")
        output = out
    
    # Flatten input along the specified dimension
    if dim == 0:
        input_flat = input.view(-1, *shape[1:])
    elif dim == input.dim() - 1:
        input_flat = input.view(*shape[:-1], -1)
    else:
        # For middle dimensions, we need to reshape
        input_flat = input.view(*shape[:dim], -1, *shape[dim+1:])
    
    # Compute logsumexp for each slice
    for i in range(input_flat.shape[0]):
        if dim == 0:
            input_slice = input_flat[i]
        elif dim == input.dim() - 1:
            input_slice = input_flat[..., i]
        else:
            # For middle dimensions, we need to handle indexing
            input_slice = input_flat
            
        # Flatten the slice
        flat_slice = input_slice.view(-1)
        
        # Compute logsumexp
        if len(flat_slice) > 0:
            # Use a simple approach for now
            max_val = flat_slice.max()
            sum_exp = (flat_slice - max_val).exp().sum()
            result = max_val + sum_exp.log()
            
            # Store result
            if out is not None:
                out[i] = result
            else:
                output[i] = result
    
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

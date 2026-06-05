import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(
    input_ptr, 
    output_ptr, 
    num_elements, 
    dim_size, 
    stride_input, 
    stride_output,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    input_ptrs = input_ptr + offsets * stride_input
    input_vals = tl.load(input_ptrs, mask=mask)
    
    # Find max value for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute sum of exponentials
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute log of sum of exponentials
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Store result
    output_ptrs = output_ptr + pid * stride_output
    tl.store(output_ptrs, log_sum_exp)

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    if dim < 0:
        dim = input.dim() + dim
    
    if out is not None:
        output = out
    else:
        output_shape = list(input.shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    if input.numel() == 0:
        return output
    
    # Flatten the tensor for processing
    input_flat = input.flatten()
    output_flat = output.flatten()
    
    # Calculate block size and number of blocks
    BLOCK_SIZE = 1024
    num_blocks = (input_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    grid = (num_blocks,)
    logsumexp_kernel[grid](
        input_flat,
        output_flat,
        input_flat.numel(),
        input.shape[dim],
        1,
        1,
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

import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size, output_size,
    alpha, dim_size, keepdim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < output_size
    
    input_offsets = offset + tl.arange(0, BLOCK_SIZE)
    other_offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    other_vals = tl.load(other_ptr + other_offsets, mask=mask)
    
    result = input_vals + alpha * other_vals
    tl.store(output_ptr + input_offsets, result, mask=mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
        if isinstance(other, torch.Tensor):
            other = other.to(dtype)
    
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    
    # Broadcasting
    input_shape = input.shape
    other_shape = other.shape
    
    # Compute output shape
    if dim is None:
        output_shape = ()
    else:
        if isinstance(dim, int):
            dim = (dim,)
        output_shape = tuple(s for i, s in enumerate(input_shape) if i not in dim)
        if keepdim:
            output_shape = tuple(s if i not in dim else 1 for i, s in enumerate(input_shape))
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    
    # Handle scalar case
    if input.numel() == 1:
        result = input + alpha * other
        if dim is None:
            output = result.sum()
        else:
            output = result.mean(dim=dim, keepdim=keepdim)
        return output
    
    # For multi-element tensors, use Triton kernel
    if dim is None:
        # Compute total elements
        total_elements = input.numel()
        BLOCK_SIZE = 1024
        num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Flatten tensors for kernel
        input_flat = input.flatten()
        other_flat = other.flatten()
        output_flat = output.flatten()
        
        add_mean_kernel[(num_blocks,)](
            input_flat, other_flat, output_flat,
            total_elements, total_elements, total_elements,
            alpha, 1, False, BLOCK_SIZE
        )
        
        # Compute mean
        output = output_flat.mean()
    else:
        # For specific dimensions, we need to handle reduction properly
        # This is a simplified version - full implementation would be more complex
        result = input + alpha * other
        output = result.mean(dim=dim, keepdim=keepdim)
    
    return output

##################################################################################################################################################



import torch

def test_add_mean():
    results = {}

    # Test case 1: Basic addition and mean with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_mean(input1, other1)

    # Test case 2: Addition with scalar other and non-default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = add_mean(input2, other2, alpha=2)

    # Test case 3: Addition with mean along a specific dimension
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = add_mean(input3, other3, dim=0)

    # Test case 4: Addition with mean and keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = add_mean(input4, other4, dim=1, keepdim=True)

    return results

test_results = test_add_mean()

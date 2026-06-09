import torch
import triton
import triton.language as tl

def permute_copy(input, dims):
    # Validate input
    if not torch.is_tensor(input):
        raise TypeError("input must be a torch.Tensor")
    
    # Validate dims
    if not isinstance(dims, (tuple, list)):
        raise TypeError("dims must be a tuple or list")
    
    # Check if dims is valid
    if len(dims) != input.dim():
        raise ValueError("dims must have the same length as input dimensions")
    
    # Check if all dimensions are present
    if sorted(dims) != list(range(input.dim())):
        raise ValueError("dims must contain each dimension index exactly once")
    
    # Create output tensor with correct shape
    new_shape = tuple(input.shape[i] for i in dims)
    out = torch.empty(new_shape, dtype=input.dtype, device=input.device)
    
    # Handle special case of no permutation
    if dims == tuple(range(input.dim())):
        out.copy_(input)
        return out
    
    # For small tensors, use PyTorch's implementation
    if input.numel() < 1024:
        return input.permute(dims)
    
    # For larger tensors, use Triton kernel
    # We'll use a simple approach: create a mapping from output indices to input indices
    # and copy elements accordingly
    
    # Create inverse permutation to map output indices to input indices
    inv_dims = [0] * len(dims)
    for i, dim in enumerate(dims):
        inv_dims[dim] = i
    
    # Flatten both tensors
    input_flat = input.flatten()
    out_flat = out.flatten()
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _permute_copy_kernel[grid](
        input_flat, out_flat, 
        input.stride(0) if input.dim() > 0 else 1,
        out.stride(0) if out.dim() > 0 else 1,
        n,
        BLOCK=block
    )
    
    return out

@triton.jit
def _permute_copy_kernel(input_ptr, output_ptr, input_stride, output_stride, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input element
    input_val = tl.load(input_ptr + offsets * input_stride, mask=mask)
    
    # Store to output
    tl.store(output_ptr + offsets * output_stride, input_val, mask=mask)
##################################################################################################################################################



import torch

def test_permute_copy():
    results = {}

    # Test case 1: Simple 2D tensor permutation
    tensor_2d = torch.tensor([[1, 2], [3, 4]], device='cuda')
    results["test_case_1"] = permute_copy(tensor_2d, [1, 0])

    # Test case 2: 3D tensor permutation
    tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device='cuda')
    results["test_case_2"] = permute_copy(tensor_3d, [2, 0, 1])

    # Test case 3: Permutation with no change
    tensor_no_change = torch.tensor([1, 2, 3, 4], device='cuda')
    results["test_case_3"] = permute_copy(tensor_no_change, [0])

    # Test case 4: Higher dimensional tensor permutation
    tensor_4d = torch.rand((2, 3, 4, 5), device='cuda')
    results["test_case_4"] = permute_copy(tensor_4d, [3, 2, 1, 0])

    return results

test_results = test_permute_copy()

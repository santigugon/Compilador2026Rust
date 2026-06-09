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
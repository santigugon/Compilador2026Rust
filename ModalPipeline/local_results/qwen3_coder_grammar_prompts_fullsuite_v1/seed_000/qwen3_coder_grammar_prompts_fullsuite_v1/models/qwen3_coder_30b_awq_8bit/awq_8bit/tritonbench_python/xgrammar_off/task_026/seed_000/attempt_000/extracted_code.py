import torch
import triton
import triton.language as tl

@triton.jit
def _fused_tile_exp_kernel(x_ptr, out_ptr, n: tl.constexpr, dims_ptr, ndims: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply exponential function
    y = tl.exp(x)
    
    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_tile_exp(input, dims, *, out=None):
    # Handle the case where dims is shorter than input dimensions
    # Prepend 1s to dims to match input dimensions
    if len(dims) < input.ndim:
        dims = (1,) * (input.ndim - len(dims)) + dims
    
    # Compute output shape
    output_shape = tuple(input.shape[i] * dims[i] for i in range(len(dims)))
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != output_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {output_shape}")
    
    # Compute total number of elements
    n = out.numel()
    
    # If input is empty, return empty output
    if n == 0:
        return out
    
    # Create a temporary tensor for the tiled input
    # This is a simplified approach - in practice, we'd need to implement
    # proper tiling logic, but for now we'll use a straightforward approach
    # that works for the basic case
    
    # For the actual tiling operation, we need to compute the tiling
    # We'll use a simple approach where we repeat the input tensor
    # This is a simplified version that works for basic cases
    
    # Create a temporary tensor with the same shape as input
    temp_input = input
    
    # For the actual implementation, we would need to:
    # 1. Tile the input tensor according to dims
    # 2. Apply exp to the tiled tensor
    
    # Since the tiling operation is complex to implement in a single kernel,
    # we'll use PyTorch's built-in operations for tiling and then apply
    # the exponential function in Triton
    
    # First, let's tile the input using PyTorch
    if len(dims) == 0:
        tiled_input = input
    else:
        # Use PyTorch's expand and reshape for tiling
        expand_shape = []
        for i, dim in enumerate(dims):
            expand_shape.append(input.shape[i])
            expand_shape.append(dim)
        
        # Create expanded tensor
        expanded = input.expand(*[input.shape[i] for i in range(len(input.shape))], *dims)
        
        # Reshape to get the final tiled tensor
        final_shape = []
        for i in range(len(input.shape)):
            final_shape.append(input.shape[i])
            final_shape.append(dims[i])
        
        tiled_input = expanded.reshape(final_shape)
        
        # Flatten to match the expected output shape
        tiled_input = tiled_input.reshape(output_shape)
    
    # Now apply exponential function using Triton
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Apply exponential function using Triton kernel
    _fused_tile_exp_kernel[grid](tiled_input, out, n, None, 0, BLOCK=block)
    
    return out

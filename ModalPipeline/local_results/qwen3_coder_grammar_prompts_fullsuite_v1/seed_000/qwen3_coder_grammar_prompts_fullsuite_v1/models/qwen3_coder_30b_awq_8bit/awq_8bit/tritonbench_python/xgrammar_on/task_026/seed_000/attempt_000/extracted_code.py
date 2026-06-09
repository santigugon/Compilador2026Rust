import torch
import triton
import triton.language as tl

def _ceildiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _fused_tile_exp_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    output_shape,
    input_strides,
    output_strides,
    numel_output: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel_output
    
    # Compute multi-dimensional indices for output
    output_indices = tl.zeros((BLOCK,), dtype=tl.int64)
    temp_offsets = offsets
    
    # Convert linear offset to multi-dimensional indices
    for i in range(len(output_shape) - 1, -1, -1):
        output_indices = tl.where(i == len(output_shape) - 1, temp_offsets % output_shape[i], output_indices)
        temp_offsets = temp_offsets // output_shape[i]
    
    # Compute input indices by taking modulo with input shape
    input_indices = output_indices % input_shape
    
    # Compute input linear index
    input_offset = 0
    for i in range(len(input_shape)):
        input_offset += input_indices[i] * input_strides[i]
    
    # Load input value
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Apply exponential function
    exp_val = tl.exp(input_val)
    
    # Store result
    tl.store(output_ptr + offsets, exp_val, mask=mask)


def fused_tile_exp(input, dims, *, out=None):
    # Handle scalar input
    if input.dim() == 0:
        if out is not None:
            out.copy_(torch.exp(input))
            return out
        return torch.exp(input)
    
    # Prepare dims
    if len(dims) < input.dim():
        dims = (1,) * (input.dim() - len(dims)) + dims
    
    # Compute output shape
    output_shape = tuple(input.shape[i] * dims[i] for i in range(len(dims)))
    
    # Create output tensor
    if out is not None:
        if out.shape != output_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {output_shape}")
    else:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Get input strides
    input_strides = input.stride()
    
    # Get output strides
    output_strides = out.stride()
    
    # Flatten input and output for kernel
    input_flat = input.contiguous().view(-1)
    output_flat = out.contiguous().view(-1)
    
    # Compute total number of elements in output
    numel_output = output_flat.numel()
    
    # Launch kernel
    block = 256
    grid = _ceildiv(numel_output, block)
    
    # Prepare shape and stride arrays for kernel
    input_shape = tuple(input.shape)
    output_shape = tuple(out.shape)
    
    _fused_tile_exp_kernel[grid](
        input_flat,
        output_flat,
        input_shape,
        output_shape,
        input_strides,
        output_strides,
        numel_output,
        BLOCK=block
    )
    
    return out
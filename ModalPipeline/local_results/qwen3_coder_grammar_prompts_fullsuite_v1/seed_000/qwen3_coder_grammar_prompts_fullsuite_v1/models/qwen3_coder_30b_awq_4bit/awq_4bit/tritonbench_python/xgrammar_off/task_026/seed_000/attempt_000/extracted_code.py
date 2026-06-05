import torch
import triton
import triton.language as tl

@triton.jit
def _tile_exp_kernel(x_ptr, out_ptr, x_size, out_size, dims, num_dims, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_size
    
    # Calculate the output index
    out_idx = offsets
    
    # Calculate the corresponding input index
    input_idx = 0
    temp_idx = out_idx
    stride = 1
    
    # Reverse the dims to calculate input index
    for i in range(num_dims - 1, -1, -1):
        dim_size = dims[i]
        if i == num_dims - 1:
            input_idx += (temp_idx % dim_size) * stride
        else:
            input_idx += (temp_idx % dim_size) * stride
        temp_idx //= dim_size
        stride *= x_size[i]
    
    # Load input value and compute exp
    x = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_tile_exp(input, dims, *, out=None):
    # Handle case where dims is shorter than input dimensions
    if len(dims) < input.dim():
        dims = (1,) * (input.dim() - len(dims)) + tuple(dims)
    
    # Calculate output shape
    out_shape = tuple(input.shape[i] * dims[i] for i in range(len(dims)))
    
    # Create output tensor
    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != out_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {out_shape}")
    
    # Flatten input and output for easier indexing
    x_flat = input.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    
    # Calculate total elements
    x_size = input.shape
    out_size = out.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(out_size, block),)
    
    # Convert dims to a list for easier access in kernel
    dims_list = list(dims)
    
    _tile_exp_kernel[grid](x_flat, out_flat, x_size, out_size, dims_list, len(dims_list), BLOCK=block)
    
    return out

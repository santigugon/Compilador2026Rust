import torch
import triton
import triton.language as tl

def _ceildiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _tile_exp_kernel(x_ptr, out_ptr, x_size, out_size, dims_ptr, ndim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_size
    
    # Calculate output coordinates
    out_coords = tl.zeros((ndim,), dtype=tl.int32)
    temp = offsets
    for i in range(ndim - 1, -1, -1):
        out_coords[i] = temp % dims_ptr[i]
        temp = temp // dims_ptr[i]
    
    # Calculate input coordinates
    in_coords = out_coords
    for i in range(ndim):
        in_coords[i] = in_coords[i] % (x_size // dims_ptr[i])
    
    # Calculate input index
    x_idx = 0
    stride = 1
    for i in range(ndim - 1, -1, -1):
        x_idx += in_coords[i] * stride
        stride *= x_size // dims_ptr[i]
    
    x = tl.load(x_ptr + x_idx, mask=True, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

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
    
    # Calculate output size
    out_size = 1
    for i in range(input.dim()):
        out_size *= dims[i]
    
    # Create output tensor
    if out is None:
        out = torch.empty(out_size, dtype=input.dtype, device=input.device)
    else:
        if out.numel() != out_size:
            raise ValueError(f"Output tensor size {out.numel()} does not match expected size {out_size}")
    
    # Flatten input for easier indexing
    x_flat = input.flatten()
    
    # Prepare dims tensor
    dims_tensor = torch.tensor(dims, dtype=torch.int32, device=input.device)
    
    # Launch kernel
    block = 256
    grid = _ceildiv(out_size, block)
    
    _tile_exp_kernel[grid](x_flat, out, input.numel(), out_size, dims_tensor, input.dim(), BLOCK=block)
    
    # Reshape output to match expected shape
    out_shape = []
    for i in range(input.dim()):
        out_shape.append(input.shape[i] * dims[i])
    out = out.view(out_shape)
    
    return out
##################################################################################################################################################



import torch

def test_fused_tile_exp():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    dims1 = (2,)
    results["test_case_1"] = fused_tile_exp(input1, dims1)

    # Test case 2: Tiling with multiple dimensions
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    dims2 = (2, 3)
    results["test_case_2"] = fused_tile_exp(input2, dims2)

    # Test case 3: Tiling with fewer dimensions specified
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    dims3 = (2,)
    results["test_case_3"] = fused_tile_exp(input3, dims3)

    # Test case 4: Using the out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    dims4 = (2,)
    out_tensor = torch.empty((2, 3), device='cuda')
    results["test_case_4"] = fused_tile_exp(input4, dims4, out=out_tensor)

    return results

test_results = test_fused_tile_exp()

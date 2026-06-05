import torch
import triton
import triton.language as tl

@triton.jit
def fused_tile_exp_kernel(
    input_ptr, 
    output_ptr, 
    input_shape, 
    output_shape,
    num_elements,
    tile_dims,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply exponential function
    exp_elements = tl.exp(input_elements)
    
    # Store result
    tl.store(output_ptr + offsets, exp_elements, mask=mask)

def fused_tile_exp(input, dims, *, out=None):
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not isinstance(dims, tuple):
        raise TypeError("dims must be a tuple")
    
    # Handle tiling
    input_shape = input.shape
    expanded_dims = (1,) * (len(dims) - len(input_shape)) + dims
    output_shape = tuple(s * d for s, d in zip(input_shape, expanded_dims))
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != output_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {output_shape}")
    
    # Prepare for kernel launch
    num_elements = out.numel()
    BLOCK_SIZE = 1024
    grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_tile_exp_kernel[grid_size](
        input.data_ptr(),
        out.data_ptr(),
        input_shape,
        output_shape,
        num_elements,
        expanded_dims,
        BLOCK_SIZE
    )
    
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

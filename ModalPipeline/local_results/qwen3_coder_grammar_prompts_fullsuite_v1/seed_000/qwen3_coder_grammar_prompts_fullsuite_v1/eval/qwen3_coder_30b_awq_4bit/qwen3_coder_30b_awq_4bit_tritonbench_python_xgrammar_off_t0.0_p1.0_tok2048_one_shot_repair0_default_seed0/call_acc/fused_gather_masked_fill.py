import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    input_ptr, 
    index_ptr, 
    mask_ptr, 
    output_ptr,
    input_shape, 
    index_shape, 
    mask_shape,
    value,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input, index, and mask
    input_ptrs = input_ptr + offsets
    index_ptrs = index_ptr + offsets
    mask_ptrs = mask_ptr + offsets
    
    # Gather operation
    indices = tl.load(index_ptrs, mask=mask)
    input_vals = tl.load(input_ptrs, mask=mask)
    
    # Apply mask and fill with value
    mask_vals = tl.load(mask_ptrs, mask=mask)
    output_vals = tl.where(mask_vals, value, input_vals)
    
    # Store result
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, output_vals, mask=mask)

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Validate inputs
    if input.dim() != index.dim():
        raise ValueError("input and index must have the same number of dimensions")
    
    if not mask.shape == index.shape:
        raise ValueError("mask and index must have the same shape")
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")
    
    # Handle device placement
    device = input.device
    if device.type != 'cuda':
        raise ValueError("Only CUDA devices are supported")
    
    # Flatten tensors for kernel execution
    input_flat = input.contiguous().view(-1)
    index_flat = index.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    
    # Calculate number of elements
    num_elements = input_flat.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_gather_masked_fill_kernel[grid_size,](
        input_flat,
        index_flat,
        mask_flat,
        out_flat,
        input.shape,
        index.shape,
        mask.shape,
        value,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch

def test_fused_gather_masked_fill():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.tensor([[1, 2], [3, 4]], device='cuda')
    index1 = torch.tensor([[0, 1], [1, 0]], device='cuda')
    mask1 = torch.tensor([[True, False], [False, True]], device='cuda')
    value1 = -1.0
    results["test_case_1"] = fused_gather_masked_fill(input1, 1, index1, mask1, value1)

    # Test case 2: Different dimension
    input2 = torch.tensor([[5, 6, 7], [8, 9, 10]], device='cuda')
    index2 = torch.tensor([[0, 2], [1, 0]], device='cuda')
    mask2 = torch.tensor([[False, True], [True, False]], device='cuda')
    value2 = 0.0
    results["test_case_2"] = fused_gather_masked_fill(input2, 1, index2, mask2, value2)

    # Test case 3: Sparse gradient
    input3 = torch.tensor([[11, 12], [13, 14]], device='cuda')
    index3 = torch.tensor([[1, 0], [0, 1]], device='cuda')
    mask3 = torch.tensor([[True, True], [False, False]], device='cuda')
    value3 = 99.0
    results["test_case_3"] = fused_gather_masked_fill(input3, 1, index3, mask3, value3, sparse_grad=True)

    # Test case 4: Larger tensor
    input4 = torch.tensor([[15, 16, 17, 18], [19, 20, 21, 22]], device='cuda')
    index4 = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], device='cuda')
    mask4 = torch.tensor([[False, False, True, True], [True, False, False, True]], device='cuda')
    value4 = -5.0
    results["test_case_4"] = fused_gather_masked_fill(input4, 1, index4, mask4, value4)

    return results

test_results = test_fused_gather_masked_fill()

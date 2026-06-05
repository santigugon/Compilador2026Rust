import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    input_ptr, index_ptr, mask_ptr, output_ptr,
    input_shape, index_shape, mask_shape,
    dim, value, 
    input_strides, index_strides, mask_strides, output_strides,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load indices
    index_offsets = tl.multiple_of(offsets, 1)
    index_val = tl.load(index_ptr + index_offsets, mask=mask)
    
    # Compute gather indices
    gather_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for i in range(len(input_shape)):
        if i == dim:
            gather_indices = index_val
        else:
            # For simplicity, assuming 1D case for now
            gather_indices = tl.broadcast_to(index_val, (BLOCK_SIZE,))
    
    # Gather values
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for i in range(len(input_shape)):
        if i == dim:
            input_offsets += index_val * input_strides[i]
        else:
            input_offsets += tl.zeros((BLOCK_SIZE,), dtype=tl.int64) * input_strides[i]
    
    input_val = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Load mask
    mask_val = tl.load(mask_ptr + offsets, mask=mask)
    
    # Apply masked fill
    result = tl.where(mask_val, value, input_val)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Validate inputs
    assert input.dim() == index.dim(), "input and index must have the same number of dimensions"
    assert mask.shape == out.shape, "mask shape must match output shape"
    
    # Compute output shape
    output_shape = list(input.shape)
    for i in range(len(output_shape)):
        if i != dim:
            output_shape[i] = index.shape[i]
    
    # Flatten tensors for kernel execution
    input_flat = input.flatten()
    index_flat = index.flatten()
    mask_flat = mask.flatten()
    out_flat = out.flatten()
    
    # Compute strides
    input_strides = [input.stride(i) for i in range(input.dim())]
    index_strides = [index.stride(i) for i in range(index.dim())]
    mask_strides = [mask.stride(i) for i in range(mask.dim())]
    output_strides = [out.stride(i) for i in range(out.dim())]
    
    # Launch kernel
    num_elements = input_flat.numel()
    BLOCK_SIZE = 1024
    grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_gather_masked_fill_kernel[grid_size](
        input_flat, index_flat, mask_flat, out_flat,
        input.shape, index.shape, mask.shape,
        dim, value,
        input_strides, index_strides, mask_strides, output_strides,
        num_elements,
        BLOCK_SIZE
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

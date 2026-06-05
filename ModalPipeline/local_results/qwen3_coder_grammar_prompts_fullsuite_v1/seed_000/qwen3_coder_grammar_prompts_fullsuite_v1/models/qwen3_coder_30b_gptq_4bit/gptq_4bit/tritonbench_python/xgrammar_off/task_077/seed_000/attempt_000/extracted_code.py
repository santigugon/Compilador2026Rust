import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    input_ptr, index_ptr, mask_ptr, output_ptr,
    input_stride_0, input_stride_1,
    index_stride_0, index_stride_1,
    mask_stride_0, mask_stride_1,
    output_stride_0, output_stride_1,
    N, M,
    BLOCK_SIZE: tl.constexpr,
    value: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    
    for i in range(block_start, block_end):
        # Load index
        index_val = tl.load(index_ptr + i * index_stride_0)
        
        # Load input value
        input_val = tl.load(input_ptr + i * input_stride_0 + index_val * input_stride_1)
        
        # Load mask
        mask_val = tl.load(mask_ptr + i * mask_stride_0 + index_val * mask_stride_1)
        
        # Compute output
        output_val = tl.where(mask_val, value, input_val)
        
        # Store output
        tl.store(output_ptr + i * output_stride_0 + index_val * output_stride_1, output_val)

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Validate inputs
    assert input.dim() == index.dim(), "input and index must have the same number of dimensions"
    assert mask.shape == index.shape, "mask and index must have the same shape"
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "out tensor must have the same shape as input"
    
    # Get dimensions
    N, M = input.shape[0], input.shape[1]
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Set up strides
    input_stride_0 = input.stride(0)
    input_stride_1 = input.stride(1)
    index_stride_0 = index.stride(0)
    index_stride_1 = index.stride(1)
    mask_stride_0 = mask.stride(0)
    mask_stride_1 = mask.stride(1)
    output_stride_0 = out.stride(0)
    output_stride_1 = out.stride(1)
    
    # Launch kernel
    fused_gather_masked_fill_kernel[grid](
        input_ptr=input.data_ptr(),
        index_ptr=index.data_ptr(),
        mask_ptr=mask.data_ptr(),
        output_ptr=out.data_ptr(),
        input_stride_0=input_stride_0,
        input_stride_1=input_stride_1,
        index_stride_0=index_stride_0,
        index_stride_1=index_stride_1,
        mask_stride_0=mask_stride_0,
        mask_stride_1=mask_stride_1,
        output_stride_0=output_stride_0,
        output_stride_1=output_stride_1,
        N=N,
        M=M,
        BLOCK_SIZE=BLOCK_SIZE,
        value=value
    )
    
    return out

import torch
import triton
import triton.language as tl

def _get_strides(tensor):
    return tensor.stride()

def _get_shape(tensor):
    return tensor.shape

@triton.jit
def _index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape, input_strides,
    index_strides,
    other_strides,
    out_strides,
    dim_size: tl.constexpr,
    index_size: tl.constexpr,
    other_size: tl.constexpr,
    out_size: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_size
    
    # Compute the output index
    out_idx = offsets
    
    # Compute the input index
    input_idx = tl.zeros((BLOCK,), dtype=tl.int64)
    
    # Calculate strides for the output tensor
    out_strides_list = [out_strides[i] for i in range(len(out_strides))]
    
    # Calculate the index for the selected dimension
    dim_offset = 0
    for i in range(len(out_strides_list)):
        if i == dim:
            dim_offset = (out_idx // out_strides_list[i]) % dim_size
        else:
            # For other dimensions, we need to compute the correct index
            pass
    
    # Load index
    index_val = tl.load(index_ptr + dim_offset, mask=(dim_offset < index_size))
    
    # Compute input index
    input_idx = tl.zeros((BLOCK,), dtype=tl.int64)
    
    # Load input element
    input_val = tl.load(input_ptr + index_val, mask=(index_val < dim_size))
    
    # Load other element
    other_val = tl.load(other_ptr + out_idx, mask=mask)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + out_idx, result, mask=mask)

@triton.jit
def _index_select_eq_kernel_simple(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_size: tl.constexpr,
    index_size: tl.constexpr,
    other_size: tl.constexpr,
    out_size: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_size
    
    # Load index
    index_val = tl.load(index_ptr + offsets, mask=mask)
    
    # Load input element
    input_val = tl.load(input_ptr + index_val, mask=mask)
    
    # Load other element
    other_val = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _index_select_eq_kernel_broadcast(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape, input_strides,
    index_strides,
    other_strides,
    out_strides,
    dim_size: tl.constexpr,
    index_size: tl.constexpr,
    other_size: tl.constexpr,
    out_size: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_size
    
    # Compute the output index
    out_idx = offsets
    
    # Load index
    index_val = tl.load(index_ptr + out_idx, mask=mask)
    
    # Load input element
    input_val = tl.load(input_ptr + index_val, mask=mask)
    
    # Load other element
    other_val = tl.load(other_ptr + out_idx, mask=mask)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + out_idx, result, mask=mask)

@triton.jit
def _index_select_eq_kernel_simple_broadcast(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_size: tl.constexpr,
    index_size: tl.constexpr,
    other_size: tl.constexpr,
    out_size: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_size
    
    # Load index
    index_val = tl.load(index_ptr + offsets, mask=mask)
    
    # Load input element
    input_val = tl.load(input_ptr + index_val, mask=mask)
    
    # Load other element
    other_val = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if not torch.is_tensor(index):
        raise TypeError("index must be a tensor")
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Get the shape and strides
    input_shape = input.shape
    input_strides = input.stride()
    index_strides = index.stride()
    other_strides = other.stride()
    
    # Compute output shape
    out_shape = list(input_shape)
    out_shape[dim] = index.shape[0]
    
    # Create output tensor
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(out_shape):
            raise ValueError("Output tensor shape does not match expected shape")
        if out.dtype != torch.bool:
            raise ValueError("Output tensor must be boolean type")
    
    # Flatten tensors for kernel execution
    input_flat = input.view(-1)
    index_flat = index.view(-1)
    other_flat = other.view(-1)
    out_flat = out.view(-1)
    
    # Get sizes
    input_size = input_flat.numel()
    index_size = index_flat.numel()
    other_size = other_flat.numel()
    out_size = out_flat.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(out_size, block),)
    
    # Use simple kernel for now
    _index_select_eq_kernel_simple[grid](
        input_flat, index_flat, other_flat, out_flat,
        input_size, index_size, other_size, out_size,
        dim, BLOCK=block
    )
    
    return out
##################################################################################################################################################



import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()

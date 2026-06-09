import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List

def _get_contracted_dims(dims, a_ndim, b_ndim):
    if isinstance(dims, int):
        return list(range(a_ndim - dims, a_ndim)), list(range(dims))
    elif isinstance(dims, tuple) and len(dims) == 2:
        return dims[0], dims[1]
    elif isinstance(dims, list) and len(dims) == 2:
        return dims[0], dims[1]
    else:
        raise ValueError("dims must be int, tuple of two lists, or list of two lists")

def _get_remaining_dims(dims, ndim):
    all_dims = list(range(ndim))
    return [d for d in all_dims if d not in dims]

def _compute_output_shape(a_shape, b_shape, a_contract_dims, b_contract_dims):
    a_remain_dims = _get_remaining_dims(a_contract_dims, len(a_shape))
    b_remain_dims = _get_remaining_dims(b_contract_dims, len(b_shape))
    return [a_shape[i] for i in a_remain_dims] + [b_shape[i] for i in b_remain_dims]

def _compute_strides(shape):
    strides = [1]
    for i in range(len(shape) - 1, 0, -1):
        strides = [shape[i] * strides[0]] + strides
    return strides

def _get_tensor_strides(tensor):
    return list(tensor.stride())

@triton.jit
def _tensordot_kernel(a_ptr, b_ptr, out_ptr, a_strides, b_strides, out_strides,
                      a_shape, b_shape, out_shape,
                      a_contract_dims, b_contract_dims,
                      a_remain_dims, b_remain_dims,
                      a_ndim, b_ndim, out_ndim,
                      contract_size: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    
    # Get thread indices
    pid = tl.program_id(0)
    
    # Compute output indices
    out_idx = pid
    out_indices = []
    temp = out_idx
    for i in range(out_ndim - 1, -1, -1):
        out_indices.append(temp % out_shape[i])
        temp //= out_shape[i]
    out_indices.reverse()
    
    # Compute a and b indices
    a_indices = []
    b_indices = []
    
    # Map output indices to a indices
    for i in range(a_ndim):
        if i in a_remain_dims:
            a_indices.append(out_indices[a_remain_dims.index(i)])
        else:
            # Find the corresponding contract dimension
            contract_idx = a_contract_dims.index(i)
            a_indices.append(out_indices[len(a_remain_dims) + contract_idx])
    
    # Map output indices to b indices
    for i in range(b_ndim):
        if i in b_remain_dims:
            b_indices.append(out_indices[b_remain_dims.index(i) + len(a_remain_dims)])
        else:
            # Find the corresponding contract dimension
            contract_idx = b_contract_dims.index(i)
            b_indices.append(out_indices[len(a_remain_dims) + contract_idx])
    
    # Compute linear indices
    a_linear_idx = 0
    b_linear_idx = 0
    
    for i in range(a_ndim):
        a_linear_idx += a_indices[i] * a_strides[i]
    
    for i in range(b_ndim):
        b_linear_idx += b_indices[i] * b_strides[i]
    
    # Compute dot product
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for i in range(contract_size):
        a_val = tl.load(a_ptr + a_linear_idx + i * a_strides[a_contract_dims[0]])
        b_val = tl.load(b_ptr + b_linear_idx + i * b_strides[b_contract_dims[0]])
        acc += a_val * b_val
    
    # Store result
    out_linear_idx = 0
    for i in range(out_ndim):
        out_linear_idx += out_indices[i] * out_strides[i]
    
    tl.store(out_ptr + out_linear_idx, acc[0])

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Handle scalar inputs
    if a.dim() == 0 or b.dim() == 0:
        return torch.tensordot(a, b, dims)
    
    # Get contracted dimensions
    a_contract_dims, b_contract_dims = _get_contracted_dims(dims, a.dim(), b.dim())
    
    # Validate dimensions
    if len(a_contract_dims) != len(b_contract_dims):
        raise ValueError("Number of contracted dimensions must be equal for both tensors")
    
    # Get remaining dimensions
    a_remain_dims = _get_remaining_dims(a_contract_dims, a.dim())
    b_remain_dims = _get_remaining_dims(b_contract_dims, b.dim())
    
    # Compute output shape
    output_shape = _compute_output_shape(a.shape, b.shape, a_contract_dims, b_contract_dims)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=a.dtype, device=a.device)
    
    # Get strides
    a_strides = _get_tensor_strides(a)
    b_strides = _get_tensor_strides(b)
    out_strides = _get_tensor_strides(out)
    
    # Get shapes
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    out_shape = list(out.shape)
    
    # Compute total contract size
    contract_size = 1
    for dim in a_contract_dims:
        contract_size *= a.shape[dim]
    
    # Launch kernel
    if contract_size > 0:
        grid = triton.cdiv(out.numel(), 256)
        BLOCK_SIZE = 256
        
        _tensordot_kernel[grid](
            a.data_ptr(), b.data_ptr(), out.data_ptr(),
            a_strides, b_strides, out_strides,
            a_shape, b_shape, out_shape,
            a_contract_dims, b_contract_dims,
            a_remain_dims, b_remain_dims,
            a.dim(), b.dim(), out.dim(),
            contract_size,
            BLOCK_SIZE
        )
    
    return out
##################################################################################################################################################



import torch
from typing import Union, List, Tuple

def test_tensordot():
    results = {}
    
    # 示例 1
    a = torch.arange(60.).reshape(3, 4, 5)
    b = torch.arange(24.).reshape(4, 3, 2)
    results["test_case_1"] = tensordot(a, b, dims=([1, 0], [0, 1]))

    # 示例 2 (在CUDA设备上)
    a = torch.randn(3, 4, 5, device='cuda')
    b = torch.randn(4, 5, 6, device='cuda')
    results["test_case_2"] = tensordot(a, b, dims=2).cpu()

    # 示例 3 (多维收缩)
    a = torch.randn(3, 5, 4, 6)
    b = torch.randn(6, 4, 5, 3)
    results["test_case_3"] = tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
    
    return results

test_results = test_tensordot()

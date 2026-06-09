import torch
import triton
import triton.language as tl
from typing import List, Tuple, Union

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Handle different input types for dims
    if isinstance(dims, int):
        # Contract last dims dimensions
        a_contract_dims = list(range(a.dim() - dims, a.dim()))
        b_contract_dims = list(range(0, dims))
    elif isinstance(dims, tuple):
        # dims is a tuple of (a_dims, b_dims)
        a_contract_dims, b_contract_dims = dims
    else:
        # dims is a list of [a_dims, b_dims]
        a_contract_dims, b_contract_dims = dims

    # Validate dimensions
    if len(a_contract_dims) != len(b_contract_dims):
        raise ValueError("Number of dimensions to contract must be equal for both tensors")

    # Check if dimensions are valid
    if any(dim < 0 or dim >= a.dim() for dim in a_contract_dims):
        raise ValueError("Invalid dimensions for tensor a")
    if any(dim < 0 or dim >= b.dim() for dim in b_contract_dims):
        raise ValueError("Invalid dimensions for tensor b")

    # Sort dimensions in descending order to avoid index shifting issues
    a_contract_dims = sorted(a_contract_dims, reverse=True)
    b_contract_dims = sorted(b_contract_dims, reverse=True)

    # Compute output shape
    a_out_shape = [a.shape[i] for i in range(a.dim()) if i not in a_contract_dims]
    b_out_shape = [b.shape[i] for i in range(b.dim()) if i not in b_contract_dims]
    output_shape = a_out_shape + b_out_shape

    # Create output tensor
    out = torch.empty(output_shape, dtype=a.dtype, device=a.device)

    # Handle scalar result case
    if len(output_shape) == 0:
        # For scalar result, we can use a simple reduction
        return _tensordot_scalar(a, b, a_contract_dims, b_contract_dims)

    # For non-scalar result, we need to compute the generalized matrix product
    # Reshape tensors for computation
    a_reshaped = _reshape_for_tensordot(a, a_contract_dims)
    b_reshaped = _reshape_for_tensordot(b, b_contract_dims)

    # Compute the matrix multiplication
    out_reshaped = torch.matmul(a_reshaped, b_reshaped)
    
    # Reshape back to output shape
    out = out_reshaped.view(output_shape)
    return out

@triton.jit
def _tensordot_kernel(a_ptr, b_ptr, out_ptr, a_size: tl.constexpr, b_size: tl.constexpr, c_size: tl.constexpr, BLOCK: tl.constexpr):
    # Compute matrix multiplication using Triton
    pid = tl.program_id(0)
    
    # Compute output indices
    row = pid // c_size
    col = pid % c_size
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute dot product
    for k in range(a_size):
        a_val = tl.load(a_ptr + row * a_size + k)
        b_val = tl.load(b_ptr + k * b_size + col)
        acc += a_val * b_val
    
    # Store result
    tl.store(out_ptr + pid, acc)

@triton.jit

@triton.jit
def _tensordot_scalar_kernel(a_ptr, b_ptr, out_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    # Compute scalar tensordot using Triton
    pid = tl.program_id(0)
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute dot product
    for i in range(size):
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        acc += a_val * b_val
    
    # Store result
    tl.store(out_ptr + pid, acc)

# Helper function to reshape tensors for tensordot

def _reshape_for_tensordot(tensor, contract_dims):
    # Remove contract dimensions and reshape to 2D
    new_shape = []
    stride = 1
    
    # Compute new shape
    for i in range(tensor.dim()):
        if i not in contract_dims:
            new_shape.append(tensor.shape[i])
        else:
            stride *= tensor.shape[i]
    
    # Add the contracted dimension
    new_shape.append(stride)
    
    # Reshape tensor
    return tensor.view(new_shape)

# Helper function for scalar tensordot

def _tensordot_scalar(a, b, a_contract_dims, b_contract_dims):
    # For scalar result, compute the dot product directly
    # Flatten the tensors to 1D
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    # Compute dot product
    result = torch.dot(a_flat, b_flat)
    return result
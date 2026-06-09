import torch
import triton
import triton.language as tl

def matmul(input, other, *, out=None):
    # Handle the case where one or both tensors are 1D
    if input.dim() == 1 and other.dim() == 1:
        # 1D dot product
        if out is not None:
            raise ValueError("out parameter not supported for 1D dot product")
        return torch.dot(input, other)
    
    # For 1D and 2D cases, we need to handle broadcasting
    if input.dim() == 1:
        input = input.unsqueeze(0)
    if other.dim() == 1:
        other = other.unsqueeze(1)
    
    # For batched operations, we need to handle the batch dimensions
    # Get the batch dimensions
    batch_dims_input = input.shape[:-2]
    batch_dims_other = other.shape[:-2]
    
    # Check if batch dimensions are compatible
    batch_shape = []
    i, j = len(batch_dims_input) - 1, len(batch_dims_other) - 1
    while i >= 0 and j >= 0:
        if batch_dims_input[i] == batch_dims_other[j]:
            batch_shape.append(batch_dims_input[i])
        elif batch_dims_input[i] == 1:
            batch_shape.append(batch_dims_other[j])
        elif batch_dims_other[j] == 1:
            batch_shape.append(batch_dims_input[i])
        else:
            raise ValueError("Incompatible batch dimensions")
        i -= 1
        j -= 1
    
    # Handle remaining batch dimensions
    while i >= 0:
        batch_shape.append(batch_dims_input[i])
        i -= 1
    while j >= 0:
        batch_shape.append(batch_dims_other[j])
        j -= 1
    
    batch_shape.reverse()
    
    # Get the matrix dimensions
    m, k = input.shape[-2], input.shape[-1]
    k2, n = other.shape[-2], other.shape[-1]
    
    if k != k2:
        raise ValueError(f"Incompatible matrix dimensions: {input.shape} and {other.shape}")
    
    # Create the output shape
    output_shape = batch_shape + [m, n]
    
    # If out is provided, check if it matches the expected shape
    if out is not None:
        if out.shape != tuple(output_shape):
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {tuple(output_shape)}")
    else:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # For 2D matrices, we can use a specialized kernel
    if input.dim() == 2 and other.dim() == 2:
        _matmul_2d_kernel(input, other, out)
    else:
        # For higher dimensional tensors, we need to iterate over batch dimensions
        # This is a simplified approach - in practice, we'd want to optimize this
        # by reshaping and using the 2D kernel
        input_reshaped = input.view(-1, m, k)
        other_reshaped = other.view(-1, k, n)
        out_reshaped = out.view(-1, m, n)
        
        for i in range(input_reshaped.shape[0]):
            _matmul_2d_kernel(input_reshaped[i], other_reshaped[i], out_reshaped[i])
    
    return out

@triton.jit
def _matmul_2d_kernel(input_ptr, other_ptr, out_ptr, m, n, k, stride_input_m, stride_input_k,
                      stride_other_k, stride_other_n, stride_out_m, stride_out_n, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offset for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for boundary conditions
    mask_m = offs_m < m
    mask_n = offs_n < n
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_start in range(0, k, BLOCK_K):
        # Load tiles
        input_tile = tl.load(input_ptr + (offs_m[:, None] * stride_input_m + 
                                          offs_k[None, :] * stride_input_k),
                             mask=(mask_m[:, None] & (offs_k[None, :] < k - k_start)),
                             other=0.0)
        other_tile = tl.load(other_ptr + (offs_k[:, None] * stride_other_k + 
                                          offs_n[None, :] * stride_other_n),
                             mask=((offs_k[:, None] < k - k_start) & mask_n[None, :]),
                             other=0.0)
        
        # Perform matrix multiplication
        acc += tl.dot(input_tile, other_tile)
    
    # Store result
    out_tile = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + (offs_m[:, None] * stride_out_m + 
                        offs_n[None, :] * stride_out_n),
             out_tile,
             mask=(mask_m[:, None] & mask_n[None, :]))

# Helper function to compute the kernel grid
@triton.jit
def _matmul_2d_kernel_grid(m, n, k, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    return num_pid_m * num_pid_n

# Helper function to perform 2D matrix multiplication
def _matmul_2d_kernel(input, other, out):
    # Get dimensions
    m, k = input.shape
    k2, n = other.shape
    
    if k != k2:
        raise ValueError(f"Incompatible matrix dimensions: {input.shape} and {other.shape}")
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Create the grid
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    
    # Launch kernel
    _matmul_2d_kernel[grid](input, other, out, m, n, k,
                           input.stride(0), input.stride(1),
                           other.stride(0), other.stride(1),
                           out.stride(0), out.stride(1),
                           BLOCK_M, BLOCK_N, BLOCK_K)
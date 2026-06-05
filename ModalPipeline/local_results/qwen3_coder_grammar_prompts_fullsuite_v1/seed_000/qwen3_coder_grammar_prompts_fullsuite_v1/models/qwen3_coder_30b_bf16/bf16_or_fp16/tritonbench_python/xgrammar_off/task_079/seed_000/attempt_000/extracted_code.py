import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(mat1_ptr, mat2_ptr, input_ptr, out_ptr, 
                  m: tl.constexpr, n: tl.constexpr, p: tl.constexpr,
                  stride_mat1_m: tl.constexpr, stride_mat1_n: tl.constexpr,
                  stride_mat2_n: tl.constexpr, stride_mat2_p: tl.constexpr,
                  stride_input_m: tl.constexpr, stride_input_p: tl.constexpr,
                  stride_out_m: tl.constexpr, stride_out_p: tl.constexpr,
                  beta: tl.constexpr, alpha: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the starting indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for boundaries
    mask_m = offs_m < m
    mask_n = offs_n < p
    mask_k = offs_k < n
    
    # Load input matrix
    input_block = tl.load(input_ptr + 
                         offs_m[:, None] * stride_input_m + 
                         offs_n[None, :] * stride_input_p,
                         mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Perform matrix multiplication
    for k in range(0, n, BLOCK_K):
        # Load blocks of mat1 and mat2
        mat1_block = tl.load(mat1_ptr + 
                            offs_m[:, None] * stride_mat1_m + 
                            (k + offs_k[None, :]) * stride_mat1_n,
                            mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        
        mat2_block = tl.load(mat2_ptr + 
                            (k + offs_k[:, None]) * stride_mat2_n + 
                            offs_n[None, :] * stride_mat2_p,
                            mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        
        # Accumulate the product
        acc += tl.dot(mat1_block, mat2_block)
    
    # Apply scaling factors
    result = alpha * acc + beta * input_block
    
    # Store the result
    tl.store(out_ptr + 
             offs_m[:, None] * stride_out_m + 
             offs_n[None, :] * stride_out_p,
             result, mask=(mask_m[:, None] & mask_n[None, :]))

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Ensure inputs are contiguous for easier handling
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    input = input.contiguous()
    
    # Get dimensions
    m, n = mat1.shape
    n2, p = mat2.shape
    
    # Validate dimensions
    if n != n2:
        raise ValueError(f"mat1 and mat2 cannot be multiplied: {m}x{n} and {n2}x{p}")
    
    # Compute output shape
    out_shape = (m, p)
    
    # Handle output tensor
    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != out_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected {out_shape}")
    
    # Handle scalar beta and alpha
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=input.device)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=input.device)
    
    # Ensure input is broadcastable with output
    if input.shape != out_shape:
        # Try to broadcast input to match output shape
        try:
            input = input.expand(out_shape)
        except RuntimeError:
            raise ValueError(f"input tensor shape {input.shape} is not broadcastable with output shape {out_shape}")
    
    # Set up block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(m, BLOCK_M)
    grid_n = triton.cdiv(p, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Get strides
    stride_mat1_m, stride_mat1_n = mat1.stride()
    stride_mat2_n, stride_mat2_p = mat2.stride()
    stride_input_m, stride_input_p = input.stride()
    stride_out_m, stride_out_p = out.stride()
    
    # Launch kernel
    _addmm_kernel[grid](
        mat1_ptr=mat1.data_ptr(),
        mat2_ptr=mat2.data_ptr(),
        input_ptr=input.data_ptr(),
        out_ptr=out.data_ptr(),
        m=m, n=n, p=p,
        stride_mat1_m=stride_mat1_m, stride_mat1_n=stride_mat1_n,
        stride_mat2_n=stride_mat2_n, stride_mat2_p=stride_mat2_p,
        stride_input_m=stride_input_m, stride_input_p=stride_input_p,
        stride_out_m=stride_out_m, stride_out_p=stride_out_p,
        beta=beta.item() if beta.numel() == 1 else beta,
        alpha=alpha.item() if alpha.numel() == 1 else alpha,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return out

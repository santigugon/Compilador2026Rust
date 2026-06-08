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
    
    # Initialize accumulator for mat1 @ mat2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute mat1 @ mat2
    for k in range(0, n, BLOCK_K):
        # Load mat1 block
        mat1_mask = (offs_m[:, None] < m) & (offs_k[None, :] < n)
        mat1 = tl.load(mat1_ptr + offs_m[:, None] * stride_mat1_m + 
                       offs_k[None, :] * stride_mat1_n, mask=mat1_mask, other=0.0)
        
        # Load mat2 block
        mat2_mask = (offs_k[:, None] < n) & (offs_n[None, :] < p)
        mat2 = tl.load(mat2_ptr + offs_k[:, None] * stride_mat2_n + 
                       offs_n[None, :] * stride_mat2_p, mask=mat2_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(mat1, mat2)
    
    # Compute output: alpha * (mat1 @ mat2) + beta * input
    out = acc * alpha
    
    # Load input if beta != 0
    if beta != 0.0:
        input_mask = (offs_m[:, None] < m) & (offs_n[None, :] < p)
        input = tl.load(input_ptr + offs_m[:, None] * stride_input_m + 
                        offs_n[None, :] * stride_input_p, mask=input_mask, other=0.0)
        out = out + beta * input
    
    # Store result
    out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < p)
    tl.store(out_ptr + offs_m[:, None] * stride_out_m + 
             offs_n[None, :] * stride_out_p, out, mask=out_mask)

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Ensure inputs are contiguous for easier handling
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    input = input.contiguous()
    
    # Get dimensions
    m, k = mat1.shape
    k2, p = mat2.shape
    
    # Check dimensions
    if k != k2:
        raise ValueError(f"mat1 and mat2 cannot be multiplied: {m}x{k} and {k2}x{p}")
    
    # Check if input is broadcastable with output
    if input.shape != (m, p):
        # Try to broadcast input to (m, p)
        try:
            input = input.expand(m, p)
        except RuntimeError:
            raise ValueError(f"input shape {input.shape} is not broadcastable to ({m}, {p})")
    
    # Create output tensor
    if out is None:
        out = torch.empty(m, p, dtype=mat1.dtype, device=mat1.device)
    else:
        if out.shape != (m, p):
            raise ValueError(f"out tensor must have shape ({m}, {p}), got {out.shape}")
    
    # Launch kernel
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    grid_m = triton.cdiv(m, BLOCK_M)
    grid_n = triton.cdiv(p, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Get strides
    stride_mat1_m, stride_mat1_k = mat1.stride()
    stride_mat2_k, stride_mat2_p = mat2.stride()
    stride_input_m, stride_input_p = input.stride()
    stride_out_m, stride_out_p = out.stride()
    
    _addmm_kernel[grid](
        mat1_ptr=mat1.data_ptr(),
        mat2_ptr=mat2.data_ptr(),
        input_ptr=input.data_ptr(),
        out_ptr=out.data_ptr(),
        m=m, n=k, p=p,
        stride_mat1_m=stride_mat1_m, stride_mat1_n=stride_mat1_k,
        stride_mat2_n=stride_mat2_k, stride_mat2_p=stride_mat2_p,
        stride_input_m=stride_input_m, stride_input_p=stride_input_p,
        stride_out_m=stride_out_m, stride_out_p=stride_out_p,
        beta=beta, alpha=alpha,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return out

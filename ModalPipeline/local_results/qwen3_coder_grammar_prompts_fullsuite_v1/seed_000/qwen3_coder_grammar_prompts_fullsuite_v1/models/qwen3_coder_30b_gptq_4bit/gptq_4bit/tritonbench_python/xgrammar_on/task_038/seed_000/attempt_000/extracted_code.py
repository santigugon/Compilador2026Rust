import torch
import triton
import triton.language as tl

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # Get dimensions
    batch_shape = input.shape[:-2]
    N, D_in = input.shape[-2], weight1.shape[0]
    D_k, D_out = weight1.shape[1], weight2.shape[1]
    
    # Reshape input for batch processing
    input_reshaped = input.view(-1, N, D_in)
    batch_size = input_reshaped.shape[0]
    
    # Output tensor
    if out is None:
        out = torch.empty_like(input_reshaped)
    else:
        assert out.shape == input_reshaped.shape, "Output tensor shape must match input shape"
    
    # Compute Z1 = input @ weight1
    Z1 = torch.empty(batch_size, N, D_k, dtype=input.dtype, device=input.device)
    _matmul_kernel[triton.cdiv(N * D_k, 256)](input_reshaped, weight1, Z1, N, D_in, D_k, 256)
    
    # Apply softmax to Z1
    Z2 = torch.empty_like(Z1)
    _softmax_kernel[triton.cdiv(N * D_k, 256)](Z1, Z2, N, D_k, 256)
    
    # Apply dropout to Z2
    Z3 = torch.empty_like(Z2)
    _dropout_kernel[triton.cdiv(N * D_k, 256)](Z2, Z3, N, D_k, dropout_p, 256)
    
    # Compute Z4 = Z3 @ weight2
    Z4 = torch.empty(batch_size, N, D_out, dtype=input.dtype, device=input.device)
    _matmul_kernel[triton.cdiv(N * D_out, 256)](Z3, weight2, Z4, N, D_k, D_out, 256)
    
    # Apply layer normalization
    Z5 = torch.empty_like(Z4)
    _layer_norm_kernel[triton.cdiv(N * D_out, 256)](Z4, residual, Z5, N, D_out, eps, 256)
    
    # Copy result to output
    out.copy_(Z5)
    
    # Reshape back to original shape
    return out.view(input.shape)

@triton.jit
def _matmul_kernel(x_ptr, w_ptr, out_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    
    # Compute row and column indices
    row = block_start // N
    col = block_start % N
    
    if row < M and col < N:
        # Compute dot product
        acc = 0.0
        for k in range(K):
            x_val = tl.load(x_ptr + row * K + k)
            w_val = tl.load(w_ptr + k * N + col)
            acc += x_val * w_val
        
        tl.store(out_ptr + row * N + col, acc)

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    
    # Compute row indices
    row = block_start // N
    
    if row < M:
        # Compute softmax
        max_val = -float('inf')
        for i in range(N):
            val = tl.load(x_ptr + row * N + i)
            max_val = tl.maximum(max_val, val)
        
        sum_val = 0.0
        for i in range(N):
            val = tl.load(x_ptr + row * N + i)
            exp_val = tl.exp(val - max_val)
            tl.store(out_ptr + row * N + i, exp_val)
            sum_val += exp_val
        
        for i in range(N):
            val = tl.load(out_ptr + row * N + i)
            tl.store(out_ptr + row * N + i, val / sum_val)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, dropout_p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    
    # Compute row and column indices
    row = block_start // N
    col = block_start % N
    
    if row < M and col < N:
        # Apply dropout
        val = tl.load(x_ptr + row * N + col)
        rand_val = tl.random.rand()  # Simple random value
        if rand_val > dropout_p:
            tl.store(out_ptr + row * N + col, val / (1.0 - dropout_p))
        else:
            tl.store(out_ptr + row * N + col, 0.0)

@triton.jit
def _layer_norm_kernel(x_ptr, residual_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    
    # Compute row indices
    row = block_start // N
    
    if row < M:
        # Compute mean and variance
        sum_val = 0.0
        for i in range(N):
            val = tl.load(x_ptr + row * N + i)
            sum_val += val
        
        mean = sum_val / N
        
        sum_sq = 0.0
        for i in range(N):
            val = tl.load(x_ptr + row * N + i)
            diff = val - mean
            sum_sq += diff * diff
        
        var = sum_sq / N
        std = tl.sqrt(var + eps)
        
        # Apply layer normalization
        for i in range(N):
            x_val = tl.load(x_ptr + row * N + i)
            residual_val = tl.load(residual_ptr + row * N + i)
            norm_val = (x_val - mean) / std
            tl.store(out_ptr + row * N + i, norm_val + residual_val)
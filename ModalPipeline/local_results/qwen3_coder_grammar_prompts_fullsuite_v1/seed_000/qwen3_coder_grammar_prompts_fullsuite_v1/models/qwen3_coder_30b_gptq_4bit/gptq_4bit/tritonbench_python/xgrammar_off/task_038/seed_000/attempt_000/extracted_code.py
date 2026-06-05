import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input = tl.load(input_ptr + offsets, mask=mask)
    input = input - tl.max(input, axis=0)
    exp_input = tl.exp(input)
    sum_exp = tl.sum(exp_input, axis=0)
    output = exp_input / sum_exp
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def dropout_kernel(
    output_ptr, input_ptr, 
    N, 
    dropout_p, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input = tl.load(input_ptr + offsets, mask=mask)
    random = tl.random_seed(0) + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    keep_prob = 1.0 - dropout_p
    dropout_mask = tl.where(random < keep_prob, 1.0, 0.0)
    output = input * dropout_mask / keep_prob
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def layer_norm_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    N, 
    eps, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input = tl.load(input_ptr + offsets, mask=mask)
    mean = tl.sum(input, axis=0) / N
    var = tl.sum((input - mean) ** 2, axis=0) / N
    std = tl.sqrt(var + eps)
    normalized = (input - mean) / std
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    output = normalized * weight + bias
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def matmul_kernel(
    output_ptr, input_ptr, weight_ptr,
    M, K, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid % (M // BLOCK_SIZE_M)
    pid_n = pid // (M // BLOCK_SIZE_M)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(input_ptr + offs_m[:, None] * K + k[None, :])
        b = tl.load(weight_ptr + k[:, None] * N + offs_n[None, :])
        accumulator += tl.dot(a, b)
    
    output = accumulator.to(tl.float32)
    tl.store(output_ptr + offs_m[:, None] * N + offs_n[None, :], output)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    batch_shape = input.shape[:-2]
    N, D_in = input.shape[-2], weight1.shape[-2]
    D_k, D_out = weight1.shape[-1], weight2.shape[-1]
    
    # Reshape input for processing
    input_reshaped = input.view(-1, N, D_in)
    batch_size = input_reshaped.shape[0]
    
    # First matmul: Z1 = input @ weight1
    Z1 = torch.empty(batch_size, N, D_k, dtype=torch.float32, device=input.device)
    grid = (triton.cdiv(N, 16) * triton.cdiv(D_k, 16),)
    matmul_kernel[grid](
        Z1, input_reshaped, weight1,
        N, D_in, D_k,
        BLOCK_SIZE_M=16, BLOCK_SIZE_K=16, BLOCK_SIZE_N=16
    )
    
    # Softmax
    Z2 = torch.empty_like(Z1)
    grid = (triton.cdiv(N * D_k, 1024),)
    softmax_kernel[grid](
        Z2, Z1,
        N * D_k,
        BLOCK_SIZE=1024
    )
    
    # Dropout
    Z3 = torch.empty_like(Z2)
    grid = (triton.cdiv(N * D_k, 1024),)
    dropout_kernel[grid](
        Z3, Z2,
        N * D_k,
        dropout_p,
        BLOCK_SIZE=1024
    )
    
    # Second matmul: Z4 = Z3 @ weight2
    Z4 = torch.empty(batch_size, N, D_out, dtype=torch.float32, device=input.device)
    grid = (triton.cdiv(N, 16) * triton.cdiv(D_out, 16),)
    matmul_kernel[grid](
        Z4, Z3, weight2,
        N, D_k, D_out,
        BLOCK_SIZE_M=16, BLOCK_SIZE_K=16, BLOCK_SIZE_N=16
    )
    
    # Add residual
    Z4 = Z4 + residual
    
    # Layer normalization
    Z5 = torch.empty_like(Z4)
    grid = (triton.cdiv(N * D_out, 1024),)
    layer_norm_kernel[grid](
        Z5, Z4, torch.ones_like(Z4), torch.zeros_like(Z4),
        N * D_out,
        eps,
        BLOCK_SIZE=1024
    )
    
    # Reshape back to original shape
    output = Z5.view(*batch_shape, N, D_out)
    
    if out is not None:
        out.copy_(output)
        return out
    return output

import torch
import triton
import triton.language as tl

def _softmax_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    # Compute softmax
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum
    tl.store(out_ptr + offsets, y, mask=mask)

def _dropout_kernel(x_ptr, out_ptr, n: tl.constexpr, dropout_p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Generate random mask
    rand = tl.random.rand(0, n)  # Use a fixed seed for reproducibility
    keep_mask = rand > dropout_p
    y = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

def _layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, mean_ptr, var_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / n
    var = tl.sum((x - mean) ** 2, axis=0) / n
    # Store mean and variance for later use
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)
    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)
    # Apply scale and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    y = x_norm * weight + bias
    tl.store(out_ptr + offsets, y, mask=mask)

def _matmul_kernel(x_ptr, w_ptr, out_ptr, m: tl.constexpr, k: tl.constexpr, n: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load x and w
    x = tl.load(x_ptr + offs_m[:, None] * k + offs_k[None, :], mask=offs_m[:, None] < m and offs_k[None, :] < k)
    w = tl.load(w_ptr + offs_k[:, None] * n + offs_n[None, :], mask=offs_k[:, None] < k and offs_n[None, :] < n)
    
    # Compute matmul
    out = tl.dot(x, w)
    
    # Store result
    tl.store(out_ptr + offs_m[:, None] * n + offs_n[None, :], out, mask=offs_m[:, None] < m and offs_n[None, :] < n)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # Get dimensions
    batch_shape = input.shape[:-2]
    N, D_in = input.shape[-2], weight1.shape[0]
    D_k = weight1.shape[1]
    D_out = weight2.shape[1]
    
    # Flatten batch dimensions
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Reshape input to (batch_size * N, D_in)
    input_flat = input.view(-1, D_in)
    
    # First matrix multiplication: Z_1 = input @ weight1
    z1 = torch.empty((batch_size * N, D_k), dtype=input.dtype, device=input.device)
    
    # Second matrix multiplication: Z_2 = Z_1 @ weight2
    z2 = torch.empty((batch_size * N, D_out), dtype=input.dtype, device=input.device)
    
    # Apply softmax to Z_1
    z1_softmax = torch.empty_like(z1)
    
    # Apply dropout to softmax result
    z1_dropout = torch.empty_like(z1_softmax)
    
    # Apply layer normalization
    z4 = torch.empty_like(input_flat)
    
    # Add residual
    output = torch.empty_like(input_flat)
    
    # Compute Z_1 = input @ weight1
    block_m = 32
    block_k = 32
    block_n = 32
    grid_m = triton.cdiv(batch_size * N, block_m)
    grid_n = triton.cdiv(D_k, block_n)
    
    # Compute Z_1
    _matmul_kernel[(grid_m, grid_n)](input_flat, weight1, z1, batch_size * N, D_in, D_k, block_m, block_k, block_n)
    
    # Compute Z_2 = Z_1 @ weight2
    grid_m = triton.cdiv(batch_size * N, block_m)
    grid_n = triton.cdiv(D_out, block_n)
    _matmul_kernel[(grid_m, grid_n)](z1, weight2, z2, batch_size * N, D_k, D_out, block_m, block_k, block_n)
    
    # Apply softmax to Z_1
    block = 256
    grid = (triton.cdiv(batch_size * N * D_k, block),)
    _softmax_kernel[grid](z1, z1_softmax, batch_size * N * D_k, block)
    
    # Apply dropout to Z_1
    _dropout_kernel[grid](z1_softmax, z1_dropout, batch_size * N * D_k, dropout_p, block)
    
    # Apply layer normalization
    # For simplicity, we'll compute layer norm on the last dimension
    # This is a simplified version - in practice, you'd need to compute mean/variance
    # across the appropriate dimensions
    z4 = torch.nn.functional.layer_norm(z2, (D_out,), eps=eps)
    
    # Add residual
    output = z4 + residual.view(-1, D_out)
    
    # Reshape output to original shape
    output = output.view(*batch_shape, N, D_out)
    
    if out is not None:
        out.copy_(output)
        return out
    return output
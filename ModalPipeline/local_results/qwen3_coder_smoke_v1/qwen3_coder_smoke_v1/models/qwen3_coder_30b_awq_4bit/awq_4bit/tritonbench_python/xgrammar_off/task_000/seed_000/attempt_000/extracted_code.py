import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(X_ptr, Y_ptr, weight_ptr, eps, N, P, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * P
    X = tl.load(X_ptr + offsets, mask=mask)
    Y = tl.load(Y_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets % P, mask=mask)
    mean = tl.sum(X * X, axis=0) / P
    inv_rms = tl.rsqrt(mean + eps)
    Y = X * inv_rms * weight
    tl.store(Y_ptr + offsets, Y, mask=mask)

@triton.jit
def gelu_kernel(X_ptr, Y_ptr, N, P, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * P

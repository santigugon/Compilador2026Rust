import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    B, N, M, P,
    normalized_shape,
    dropout_p,
    eps,
    training,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    BLOCK_SIZE_P,
    BLOCK_SIZE_BATCH,
    num_warps=4
):
    # Get the block index
    block_idx = tl.program_id(0)
    batch_idx = block_idx // (N * P // BLOCK_SIZE_N // BLOCK_SIZE_P)
    n_idx = (block_idx % (N * P // BLOCK_SIZE_N // BLOCK_SIZE_P)) * BLOCK_SIZE_N
    p_idx = (block_idx % (N * P // BLOCK_SIZE_N // BLOCK_SIZE_P)) * BLOCK_SIZE_P
    
    # Load input1 and input2
    input1_block = tl.load(input1_ptr + batch_idx * N * M + n_idx * M + tl.arange(0, BLOCK_SIZE_M)[:, None] * M + tl.arange(0, BLOCK_SIZE_K)[None, :])
    input2_block = tl.load(input2_ptr + batch_idx * M * P + tl.arange(0, BLOCK_SIZE_K)[:, None] * P + p_idx + tl.arange(0, BLOCK_SIZE_P)[None, :])
    
    # Perform batch matrix multiplication
    output_block = tl.dot(input1_block, input2_block)
    
    # Apply RMS normalization
    # Compute mean of squares
    mean_square = tl.sum(output_block * output_block, axis=1, keepdims=True) / normalized_shape
    # Add epsilon and take square root
    rms = tl.sqrt(mean_square + eps)
    # Normalize
    output_block = output_block / rms
    
    # Apply GELU activation
    output_block = 0.5 * output_block * (1 + tl.tanh(0.7978845608 * (output_block + 0.044715 * output_block * output_block * output_block)))
    
    # Apply dropout
    if training:
        # Generate random mask
        mask = tl.random.rand(BLOCK_SIZE_M, BLOCK_SIZE_P) > dropout_p
        output_block = output_block * mask / (1.0 - dropout_p)
    
    # Subtract other tensor
    other_block = tl.load(other_ptr + batch_idx * N * P + n_idx * P + tl.arange(0, BLOCK_SIZE_N)[:, None] * P + tl.arange(0, BLOCK_SIZE_P)[None, :])
    output_block = output_block - other_block
    
    # Store result
    tl.store(output_ptr + batch_idx * N * P + n_idx * P + tl.arange(0, BLOCK_SIZE_N)[:, None] * P + tl.arange(0, BLOCK_SIZE_P)[None, :], output_block)

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Validate input shapes
    assert input1.shape == (input1.shape[0], input1.shape[1], input1.shape[2])
    assert input2.shape == (input2.shape[0], input2.shape[1], input2.shape[2])
    assert other.shape == (other.shape[0], other.shape[1], other.shape[2])
    
    # Get dimensions
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Ensure other is broadcastable to (B, N, P)
    if other.shape != (B, N, P):
        other = other.expand(B, N, P)
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(B, N, P, device=input1.device, dtype=input1.dtype)
    
    # Launch kernel
    grid = (B * (N // 32) * (P // 32),)
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        input1_ptr=input1.data_ptr(),
        input2_ptr=input2.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        B=B,
        N=N,
        M=M,
        P=P,
        normalized_shape=normalized_shape,
        dropout_p=dropout_p,
        eps=eps,
        training=training,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        BLOCK_SIZE_P=32,
        num_warps=4
    )
    
    return out

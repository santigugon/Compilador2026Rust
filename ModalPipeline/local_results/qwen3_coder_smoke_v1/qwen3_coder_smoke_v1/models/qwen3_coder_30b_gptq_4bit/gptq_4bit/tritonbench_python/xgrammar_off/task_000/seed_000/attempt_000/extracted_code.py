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
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_P,
    num_warps=4
):
    # Get the block indices
    block_idx = tl.program_id(0)
    block_idy = tl.program_id(1)
    block_idz = tl.program_id(2)
    
    # Compute the starting indices for this block
    start_m = block_idx * BLOCK_SIZE_M
    start_n = block_idy * BLOCK_SIZE_N
    start_p = block_idz * BLOCK_SIZE_P
    
    # Load input1 and input2
    input1 = tl.load(input1_ptr + start_m * M + tl.arange(0, BLOCK_SIZE_M)[:, None] * M + tl.arange(0, M)[None, :])
    input2 = tl.load(input2_ptr + start_n * P + tl.arange(0, M)[:, None] *

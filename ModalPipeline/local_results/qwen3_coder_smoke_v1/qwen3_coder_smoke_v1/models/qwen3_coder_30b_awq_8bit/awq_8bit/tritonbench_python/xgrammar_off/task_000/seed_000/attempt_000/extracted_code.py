import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    rms_norm_scale_ptr,
    stride_b1, stride_n1, stride_m1,
    stride_b2, stride_m2, stride_p2,
    stride_b3, stride_n3, stride_p3,
    stride_b4, stride_n4, stride_p4,
    stride_b5, stride_n5, stride_p5,
    normalized_shape,
    dropout_p,
    training,
    eps,
    B, N, P,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load input1 and input2 for this batch and sequence
    input1_block_ptr = tl.make_block_ptr(
        base=input1_ptr,
        shape=(B, N, P),
        strides=(stride_b1, stride_n1, stride_m1),
        offsets=(batch_idx, seq_idx, 0),
        block_shape=(1,

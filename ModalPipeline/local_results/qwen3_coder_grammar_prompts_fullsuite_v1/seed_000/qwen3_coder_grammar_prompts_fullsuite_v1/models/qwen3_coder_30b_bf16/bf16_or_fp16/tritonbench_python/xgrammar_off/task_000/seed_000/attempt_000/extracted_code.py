import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    input1_stride_0, input1_stride_1, input1_stride_2,
    input2_stride_0, input2_stride_1, input2_stride_2,
    other_stride_0, other_stride_1, other_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    batch_size, seq_len, hidden_dim, output_dim,
    normalized_shape,
    dropout_p,
    training,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load input1 and input2 for this batch and sequence
    input1 = tl.load(input1_ptr + batch_idx * input1_stride_0 + seq_idx * input1_stride_1 + tl.arange(0, BLOCK_SIZE)[:, None] * input1_stride_2)
    input2 = tl.load(input2_ptr + batch_idx * input2_stride_0 + tl.arange(0, BLOCK_SIZE)[None, :] * input2_stride_1 + seq_idx * input2_stride_2)
    
    # Perform batch matrix multiplication
    result = tl.dot(input1, input2)
    
    # Apply RMS normalization
    mean = tl.sum(result * result, axis=1, keepdims=True) / normalized_shape
    rms = tl.sqrt(mean + eps)
    result = result / rms
    
    # Apply GELU activation
    result = 0.5 * result * (1 + tl.tanh(0.7978845608028654 * (result + 0.044715 * result * result * result)))
    
    # Apply dropout
    if training:
        mask = tl.rand(tl.program_id(0), tl.program_id(1)) > dropout_p
        result = result * mask / (1.0 - dropout_p)
    
    # Subtract other tensor
    other = tl.load(other_ptr + batch_idx * other_stride_0 + seq_idx * other_stride_1 + tl.arange(0, BLOCK_SIZE)[:, None] * other_stride_2)
    result = result - other
    
    # Store result
    tl.store(output_ptr + batch_idx * output_stride_0 + seq_idx * output_stride_1 + tl.arange(0, BLOCK_SIZE)[:, None] * output_stride_2, result)

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Validate input shapes
    assert input1.dim() == 3 and input2.dim() == 3
    assert input1.shape[0] == input2.shape[0] and input1.shape[2] == input2.shape[1]
    assert other.shape[-1] == input2.shape[2]
    
    # Determine output shape
    batch_size, seq_len, hidden_dim = input1.shape
    output_dim = input2.shape[2]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((batch_size, seq_len, output_dim), dtype=input1.dtype, device=input1.device)
    
    # Launch kernel
    grid = (batch_size, seq_len)
    BLOCK_SIZE = 32
    
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        input1_ptr=input1,
        input2_ptr=input2,
        other_ptr=other,
        output_ptr=out,
        input1_stride_0=input1.stride(0),
        input1_stride_1=input1.stride(1),
        input1_stride_2=input1.stride(2),
        input2_stride_0=input2.stride(0),
        input2_stride_1=input2.stride(1),
        input2_stride_2=input2.stride(2),
        other_stride_0=other.stride(0),
        other_stride_1=other.stride(1),
        other_stride_2=other.stride(2),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        output_stride_2=out.stride(2),
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        normalized_shape=normalized_shape,
        dropout_p=dropout_p,
        training=training,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

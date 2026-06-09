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
    normalized_shape, dropout_p, training, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load input1 and input2 for this batch and sequence
    input1_block_ptr = tl.make_block_ptr(
        base=input1_ptr,
        shape=(batch_size, seq_len, hidden_dim),
        strides=(input1_stride_0, input1_stride_1, input1_stride_2),
        offsets=(batch_idx, seq_idx, 0),
        block_shape=(1, 1, hidden_dim),
        order=(2, 1, 0)
    )
    input2_block_ptr = tl.make_block_ptr(
        base=input2_ptr,
        shape=(batch_size, hidden_dim, output_dim),
        strides=(input2_stride_0, input2_stride_1, input2_stride_2),
        offsets=(batch_idx, 0, 0),
        block_shape=(1, hidden_dim, output_dim),
        order=(2, 1, 0)
    )
    
    # Perform batch matrix multiplication
    input1 = tl.load(input1_block_ptr)
    input2 = tl.load(input2_block_ptr)
    bmm_result = tl.dot(input1, input2)
    
    # Apply RMS normalization
    # Compute mean of squares
    mean_sq = tl.sum(bmm_result * bmm_result, axis=1, keepdims=True) / normalized_shape
    # Add epsilon and take square root
    inv_rms = tl.rsqrt(mean_sq + eps)
    # Normalize
    normalized = bmm_result * inv_rms
    
    # Apply GELU activation
    # Using approximate GELU with tanh approximation if requested
    if approximate == 'tanh':
        gelu_result = 0.5 * normalized * (1.0 + tl.tanh(0.7978845608028654 * (normalized + 0.044715 * normalized * normalized * normalized)))
    else:
        # Standard GELU approximation using error function
        gelu_result = 0.5 * normalized * (1.0 + tl.erf(normalized / tl.sqrt(2.0)))
    
    # Apply dropout if training
    if training:
        # Generate random mask
        mask = tl.rand() > dropout_p
        gelu_result = gelu_result * mask / (1.0 - dropout_p)
    
    # Subtract other tensor
    other_block_ptr = tl.make_block_ptr(
        base=other_ptr,
        shape=(batch_size, seq_len, output_dim),
        strides=(other_stride_0, other_stride_1, other_stride_2),
        offsets=(batch_idx, seq_idx, 0),
        block_shape=(1, 1, output_dim),
        order=(2, 1, 0)
    )
    other = tl.load(other_block_ptr)
    final_result = gelu_result - other
    
    # Store result
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(batch_size, seq_len, output_dim),
        strides=(output_stride_0, output_stride_1, output_stride_2),
        offsets=(batch_idx, seq_idx, 0),
        block_shape=(1, 1, output_dim),
        order=(2, 1, 0)
    )
    tl.store(output_block_ptr, final_result)

def fused_bmm_rmsnorm_gelu_dropout_sub(
    input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None
):
    # Validate input shapes
    assert input1.dim() == 3, "input1 must be 3D tensor"
    assert input2.dim() == 3, "input2 must be 3D tensor"
    assert input1.shape[0] == input2.shape[0], "Batch sizes must match"
    assert input1.shape[2] == input2.shape[1], "Matrix dimensions must match for multiplication"
    
    batch_size, seq_len, hidden_dim = input1.shape
    _, _, output_dim = input2.shape
    
    # Ensure other is broadcastable to (B, N, P)
    if other.shape != (batch_size, seq_len, output_dim):
        # Try to broadcast other to the required shape
        try:
            other = other.expand(batch_size, seq_len, output_dim)
        except RuntimeError:
            raise ValueError("other tensor is not broadcastable to the output shape")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(batch_size, seq_len, output_dim, dtype=input1.dtype, device=input1.device)
    
    # Launch kernel
    grid = (batch_size, seq_len)
    BLOCK_SIZE = 1024
    
    # Ensure normalized_shape is an integer
    if isinstance(normalized_shape, (list, tuple)):
        normalized_shape = normalized_shape[-1] if len(normalized_shape) > 0 else 1
    elif not isinstance(normalized_shape, int):
        normalized_shape = int(normalized_shape)
    
    # Launch kernel
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        input1_ptr=input1.data_ptr(),
        input2_ptr=input2.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
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

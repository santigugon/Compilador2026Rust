import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    input1_ptr, input2_ptr, output_ptr, 
    input1_row_stride, input1_col_stride,
    input2_row_stride, input2_col_stride,
    output_row_stride, output_col_stride,
    normalized_shape, dropout_p, eps, training,
    B, N, M, P,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr
):
    # Get the batch and sequence dimensions
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load input1 and input2 for this batch and sequence
    input1_block = tl.load(input1_ptr + batch_idx * input1_row_stride + seq_idx * input1_col_stride, mask=None)
    input2_block = tl.load(input2_ptr + batch_idx * input2_row_stride + seq_idx * input2_col_stride, mask=None)
    
    # Perform batch matrix multiplication
    output_block = tl.zeros((N, P), dtype=tl.float32)
    for i in range(M):
        output_block += tl.expand_dims(input1_block[:, i], 1) * tl.expand_dims(input2_block[i, :], 0)
    
    # Apply RMS normalization
    mean = tl.sum(output_block * output_block, axis=1) / normalized_shape
    rms = tl.sqrt(mean + eps)
    output_block = output_block / rms
    
    # Apply GELU activation
    if APPROXIMATE == "tanh":
        output_block = 0.5 * output_block * (1 + tl.tanh(0.7978845608 * (output_block + 0.044715 * output_block * output_block * output_block)))
    else:
        output_block = 0.5 * output_block * (1 + tl.erf(output_block / tl.sqrt(2.0)))
    
    # Apply dropout
    if training:
        dropout_mask = tl.rand(1, 1) > dropout_p
        output_block = output_block * dropout_mask / (1.0 - dropout_p)
    
    # Store the result
    tl.store(output_ptr + batch_idx * output_row_stride + seq_idx * output_col_stride, output_block)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Validate inputs
    assert input1.dim() == 3 and input2.dim() == 3
    assert input1.size(0) == input2.size(0) and input1.size(2) == input2.size(1)
    assert isinstance(normalized_shape, (int, list, torch.Size))
    
    # Prepare dimensions
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Ensure normalized_shape is a list
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    elif isinstance(normalized_shape, torch.Size):
        normalized_shape = list(normalized_shape)
    
    # Compute output shape
    output_shape = (B, N, P)
    
    # Initialize output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input1.dtype, device=input1.device)
    else:
        assert out.shape == output_shape and out.dtype == input1.dtype and out.device == input1.device
    
    # Launch kernel
    grid = (B, N)
    BLOCK_SIZE = 32
    APPROXIMATE = 1 if approximate == "tanh" else 0
    
    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
        input1, input2, out,
        input1.stride(0), input1.stride(1),
        input2.stride(0), input2.stride(1),
        out.stride(0), out.stride(1),
        normalized_shape[0], dropout_p, eps, training,
        B, N, M, P,
        BLOCK_SIZE=BLOCK_SIZE,
        APPROXIMATE=APPROXIMATE
    )
    
    return out

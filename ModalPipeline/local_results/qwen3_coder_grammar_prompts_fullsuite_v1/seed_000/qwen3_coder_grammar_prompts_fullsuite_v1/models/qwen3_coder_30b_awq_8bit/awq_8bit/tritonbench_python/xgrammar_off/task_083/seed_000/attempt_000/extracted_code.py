import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, out_ptr, 
    dropout_mask_ptr,
    batch_size, seq_len, hidden_dim, output_dim,
    p, training, approximate,
    input1_stride_0, input1_stride_1, input1_stride_2,
    input2_stride_0, input2_stride_1, input2_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    dropout_mask_stride_0, dropout_mask_stride_1, dropout_mask_stride_2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Get the sequence index
    seq_idx = tl.program_id(1)
    
    # Get the output index
    out_idx = tl.program_id(2)
    
    # Compute the starting positions for the matrices
    input1_start = batch_idx * input1_stride_0 + seq_idx * input1_stride_1
    input2_start = batch_idx * input2_stride_0 + out_idx * input2_stride_1
    out_start = batch_idx * out_stride_0 + seq_idx * out_stride_1
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, hidden_dim, BLOCK_K):
        # Load input1 and input2
        input1 = tl.load(
            input1_ptr + input1_start + k * input1_stride_2,
            mask=(k + tl.arange(0, BLOCK_K)) < hidden_dim,
            other=0.0
        )
        input2 = tl.load(
            input2_ptr + input2_start + k * input2_stride_2,
            mask=(k + tl.arange(0, BLOCK_K)) < hidden_dim,
            other=0.0
        )
        
        # Perform matrix multiplication
        acc += tl.dot(input1, input2)
    
    # Store the result
    out = acc
    
    # Apply dropout if training
    if training:
        # Generate dropout mask
        dropout_mask = tl.load(
            dropout_mask_ptr + batch_idx * dropout_mask_stride_0 + seq_idx * dropout_mask_stride_1,
            mask=(tl.arange(0, BLOCK_N) < output_dim),
            other=0.0
        )
        # Apply dropout
        out = out * (1.0 - p) * dropout_mask
    
    # Apply GELU activation
    if approximate == 'tanh':
        # GELU with tanh approximation
        out = 0.5 * out * (1.0 + tl.tanh(0.7978845608028654 * (out + 0.044715 * out * out * out)))
    else:
        # Standard GELU
        out = 0.5 * out * (1.0 + tl.erf(out / tl.sqrt(2.0)))
    
    # Store the result
    tl.store(
        out_ptr + out_start,
        out,
        mask=(tl.arange(0, BLOCK_N) < output_dim)
    )

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    # Check input dimensions
    assert input1.dim() == 3, "input1 must be a 3D tensor"
    assert input2.dim() == 3, "input2 must be a 3D tensor"
    assert input1.size(0) == input2.size(0), "Batch sizes must match"
    assert input1.size(2) == input2.size(1), "Inner dimensions must match"
    
    # Get dimensions
    batch_size, seq_len, hidden_dim = input1.shape
    _, _, output_dim = input2.shape
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_size, seq_len, output_dim, dtype=input1.dtype, device=input1.device)
    else:
        assert out.shape == (batch_size, seq_len, output_dim), "Output tensor shape mismatch"
        assert out.dtype == input1.dtype, "Output tensor dtype mismatch"
        assert out.device == input1.device, "Output tensor device mismatch"
    
    # Create dropout mask if needed
    dropout_mask = None
    if training:
        dropout_mask = torch.rand(batch_size, seq_len, output_dim, dtype=torch.float32, device=input1.device)
        dropout_mask = (dropout_mask > p).to(torch.float32)
    
    # Launch kernel
    grid = (batch_size, seq_len, output_dim)
    block = (1, 1, 1)
    
    # Define block sizes
    BLOCK_M = 1
    BLOCK_N = 1
    BLOCK_K = 32
    
    # Launch kernel
    _bmm_dropout_gelu_kernel[grid](
        input1, input2, out, dropout_mask,
        batch_size, seq_len, hidden_dim, output_dim,
        p, training, approximate,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        dropout_mask.stride(0), dropout_mask.stride(1), dropout_mask.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return out

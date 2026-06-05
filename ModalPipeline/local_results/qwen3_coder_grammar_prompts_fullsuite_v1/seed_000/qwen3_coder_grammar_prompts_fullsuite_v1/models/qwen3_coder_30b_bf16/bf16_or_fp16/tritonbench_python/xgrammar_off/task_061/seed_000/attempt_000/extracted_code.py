import torch
import triton
import triton.language as tl

@triton.jit
def _combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
    input_n, input_din, input_dout,
    batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load weight1 (D_in, D_out)
    w1 = tl.load(weight1_ptr + tl.arange(0, input_din)[:, None] * input_dout + tl.arange(0, input_dout)[None, :])
    
    # Load input (N, D_in)
    input_offsets = batch_idx * seq_len * input_n * input_din + seq_idx * input_n * input_din + tl.arange(0, input_n)[:, None] * input_din + tl.arange(0, input_din)[None, :]
    input_data = tl.load(input_ptr + input_offsets, mask=(tl.arange(0, input_n)[:, None] < seq_len) & (tl.arange(0, input_din)[None, :] < input_din))
    
    # Matrix multiplication: input @ weight1
    activation = tl.dot(input_data, w1)
    
    # Apply sigmoid
    activation = 1.0 / (1.0 + tl.exp(-activation))
    
    # Apply tanh
    activation = 2.0 / (1.0 + tl.exp(-2.0 * activation)) - 1.0
    
    # Load weight2 (broadcastable to activation)
    w2 = tl.load(weight2_ptr + tl.arange(0, input_dout)[None, :])
    
    # Element-wise multiplication
    activation = activation * w2
    
    # Load bias (broadcastable to activation)
    b = tl.load(bias_ptr + tl.arange(0, input_dout)[None, :])
    
    # Add bias
    activation = activation + b
    
    # Store result
    out_offsets = batch_idx * seq_len * input_dout + seq_idx * input_dout + tl.arange(0, input_dout)[None, :]
    tl.store(out_ptr + out_offsets, activation, mask=tl.arange(0, input_dout)[None, :] < input_dout)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate dimensions
    batch_dims = input.shape[:-2]
    seq_len, input_din = input.shape[-2:]
    weight1_din, weight2_dout = weight1.shape
    bias_dout = bias.shape[-1] if bias.dim() > 0 else 1
    
    assert weight1_din == input_din, f"Weight1 input dimension {weight1_din} must match input dimension {input_din}"
    assert weight2_dout == input_dout, f"Weight2 output dimension {weight2_dout} must match input output dimension {input_dout}"
    assert bias_dout == input_dout, f"Bias dimension {bias_dout} must match input output dimension {input_dout}"
    
    # Compute output shape
    output_shape = batch_dims + (seq_len, input_dout)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, f"Output tensor shape {out.shape} does not match expected shape {output_shape}"
    
    # Flatten batch dimensions for kernel launch
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Launch kernel
    grid = (batch_size, seq_len)
    block_size = 256
    
    # Ensure we have enough threads for the output dimension
    if input_dout > block_size:
        block_size = input_dout
    
    _combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        seq_len, input_din, input_dout,
        batch_size, seq_len,
        BLOCK_SIZE=block_size
    )
    
    return out

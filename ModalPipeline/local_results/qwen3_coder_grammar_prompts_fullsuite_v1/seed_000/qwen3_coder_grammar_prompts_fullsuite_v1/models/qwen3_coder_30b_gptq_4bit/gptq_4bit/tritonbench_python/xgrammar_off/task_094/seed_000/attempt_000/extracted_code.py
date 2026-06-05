import torch
import triton
import triton.language as tl

@triton.jit
def dropout_sigmoid_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_samples, n_features, out_features,
    dropout_p, training,
    BLOCK_SIZE=1024
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples
    
    # Load input
    input_block = tl.load(input_ptr + offsets * n_features, mask=mask[:, None])
    
    # Linear transformation
    output_block = tl.zeros((n_samples, out_features), dtype=tl.float32)
    
    for i in range(0, n_features, BLOCK_SIZE):
        # Load weight block
        weight_offsets = tl.arange(0, BLOCK_SIZE)
        weight_mask = (i + weight_offsets) < n_features
        weight_block = tl.load(weight_ptr + (tl.arange(0, out_features)[:, None] * n_features + (i + weight_offsets)[None, :]), mask=weight_mask[None, :])
        
        # Compute linear transformation
        input_slice = input_block[:, i:i+BLOCK_SIZE]
        output_block += tl.dot(input_slice, weight_block)
    
    # Add bias
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + tl.arange(0, out_features))
        output_block += bias_block[None, :]
    
    # Apply sigmoid
    output_block = tl.sigmoid(output_block)
    
    # Apply dropout
    if training:
        dropout_mask = tl.random.rand(n_samples, out_features) > dropout_p
        output_block = output_block * dropout_mask / (1.0 - dropout_p)
    
    # Store output
    tl.store(output_ptr + offsets * out_features, output_block, mask=mask[:, None])

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    assert input.dim() >= 2
    assert weight.dim() == 2
    assert weight.size(1) == input.size(-1)
    if bias is not None:
        assert bias.dim() == 1
        assert bias.size(0) == weight.size(0)
    
    # Prepare input tensor
    input_reshaped = input.view(-1, input.size(-1))
    n_samples = input_reshaped.size(0)
    n_features = input_reshaped.size(1)
    out_features = weight.size(0)
    
    # Create output tensor
    output = torch.empty(n_samples, out_features, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    grid = (triton.cdiv(n_samples, 1024),)
    dropout_sigmoid_linear_kernel[grid](
        input_reshaped,
        weight,
        bias,
        output,
        n_samples,
        n_features,
        out_features,
        p,
        training
    )
    
    # Reshape output to match input shape
    output = output.view(*input.shape[:-1], out_features)
    
    return output

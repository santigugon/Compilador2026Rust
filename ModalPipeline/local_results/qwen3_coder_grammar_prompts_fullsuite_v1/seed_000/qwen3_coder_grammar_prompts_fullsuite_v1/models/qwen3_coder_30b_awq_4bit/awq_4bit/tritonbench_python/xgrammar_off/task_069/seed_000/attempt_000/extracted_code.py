import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, out_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    
    # Load input row
    input_row = tl.load(input_ptr + pid * input_row_stride + tl.arange(0, BLOCK_SIZE), 
                        mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Linear transformation: y = x @ weight.T + bias
    output_row = tl.zeros((out_features,), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (tl.arange(0, BLOCK_SIZE) + i) < n_cols
        weight_col = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE) + i * weight_col_stride, 
                            mask=mask, other=0.0)
        output_row += tl.sum(input_row[None, :] * weight_col[:, None], axis=0)
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, out_features), 
                       mask=tl.arange(0, out_features) < out_features, other=0.0)
        output_row += bias
    
    # Apply ReLU
    output_row = tl.where(output_row > 0, output_row, 0.0)
    
    # Layer normalization
    mean = tl.sum(output_row, axis=0) / out_features
    var = tl.sum((output_row - mean) ** 2, axis=0) / out_features
    std = tl.sqrt(var + eps)
    output_row = (output_row - mean) / std
    
    # Store result
    tl.store(output_ptr + pid * output_row_stride + tl.arange(0, out_features), 
             output_row, mask=tl.arange(0, out_features) < out_features)

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Handle default parameters
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    if not isinstance(normalized_shape, (list, tuple, torch.Size)):
        normalized_shape = [normalized_shape]
    
    # Validate input dimensions
    assert input.dim() >= 1, "Input must have at least one dimension"
    assert weight.dim() == 2, "Weight must be 2D"
    assert weight.shape[1] == input.shape[-1], "Weight column count must match input last dimension"
    
    # Prepare output shape
    output_shape = list(input.shape[:-1]) + [weight.shape[0]]
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Get dimensions
    n_rows = input.numel() // input.shape[-1]
    n_cols = input.shape[-1]
    out_features = weight.shape[0]
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_rows, 1),)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    output_ptr = out.data_ptr()
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    _fused_layer_norm_relu_linear_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        input.stride(-1), weight.stride(0), weight.stride(1),
        out.stride(-1), out.stride(-1),
        n_rows, n_cols, out_features,
        eps,
        BLOCK_SIZE
    )
    
    return out
